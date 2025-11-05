const path = require("path");
const fs = require("fs").promises;

const { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { WeaviateStore } = require('@langchain/weaviate');
const pdfjs = require("pdfjs-dist/legacy/build/pdf.js");
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { default: supabase } = require("./supabase");
const weaviateLib = require('weaviate-client').default;
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');
const { createPDFChatPrompt } = require('./pdf-chat-prompt'); // ✅ NEW: Extracted prompt
import { GoogleGenAI } from '@google/genai';
import { type } from 'os';

// PDF.js worker setup (Япон хэлний тохиргоо)
// pdfjs.GlobalWorkerOptions.workerSrc = '../../node_modules/pdfjs-dist/build/pdf.worker.mjs';

// --- Weaviate client ---
const WEAVIATE_HOST = process.env.WEAVIATE_HOST;
const WEAVIATE_API_KEY = process.env.WEAVIATE_API_KEY;


async function makeWeaviateClient() {
    // The weaviate-client package exposes helper connectToWeaviateCloud in recent versions.
    // Fallback: instantiate raw client via weaviateLib.client({ scheme, host, apiKey: new ... })
    if (typeof weaviateLib.connectToWeaviateCloud === 'function') {
        const client = await weaviateLib.connectToWeaviateCloud(WEAVIATE_HOST, {
            authCredentials: new weaviateLib.ApiKey(WEAVIATE_API_KEY),
        });
        // optional: await client.connect() if required by client version
        return client;
    } else {
        // fallback manual client creation
        const client = weaviateLib.client({
            scheme: WEAVIATE_HOST.startsWith('https') ? 'https' : 'http',
            host: WEAVIATE_HOST.replace(/^https?:\/\//, ''),
            apiKey: new weaviateLib.ApiKey(WEAVIATE_API_KEY),
        });
        return client;
    }
}

const llm = new ChatGoogleGenerativeAI({
    modelName: process.env.GEMINI_CHAT_MODEL || 'models/gemini-2.5-flash-lite',
    model: process.env.GEMINI_CHAT_MODEL || 'models/gemini-2.5-flash-lite',
    apiKey: process.env.GOOGLE_API_KEY,
    temperature: 0.1,
    // maxRetries: 2,
    // maxOutputTokens: 2048,
});

// --- Embeddings setup ---
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: 'models/gemini-embedding-001',
    apiKey: process.env.GOOGLE_API_KEY,
    batchSize: 64 // ⬅ 16-64 болгоорой, ихэнхдээ 4–5х хурдан болдог

});

/**
 * PDF-г өгсөн path-аас уншиж, text-г chunk хийж, Weaviate-д хадгалах
 * @param {string} pdfPath PDF файлын зам
 * @param {string} indexName Weaviate-д ашиглах index/collection нэр
 */


async function ingestPdfToVectorDB(pdfPath, indexName = "default_books_index") {
    const client = await makeWeaviateClient()
    try {
        console.time(`Ingestion process for ${pdfPath}`);

        await fs.access(pdfPath);
        const pdfFileName = path.basename(pdfPath);
        console.log(`Processing PDF: ${pdfFileName}`);

        // 1. PDF-г унших (Япон хэл дэмжсэн тохиргоо)
        console.time("1. Loading PDF");
        const dataBuffer = await fs.readFile(pdfPath);


        const loader = new PDFLoader(pdfPath, {
            pdfjs: () => pdfjs
        });
        const rawDocs2 = await loader.load();
        await fs.writeFile("test-docs2.json", JSON.stringify(rawDocs2, null, 2))
        // cMaps болон standard fonts зам (ABSOLUTE PATH)
        const nodeModulesPath = path.resolve(__dirname, '../../node_modules/pdfjs-dist');
        const cmapsPath = path.join(nodeModulesPath, 'cmaps').replace(/\\/g, '/') + '/';
        const fontsPath = path.join(nodeModulesPath, 'standard_fonts').replace(/\\/g, '/') + '/';

        console.log('✅ PDF.js paths:', { cmapsPath, fontsPath });

        const loadingTask = pdfjs.getDocument({
            data: new Uint8Array(dataBuffer),
            cMapUrl: cmapsPath,
            cMapPacked: true,
            standardFontDataUrl: fontsPath,
            useSystemFonts: true, // Систем дэх Япон fontуудыг ашиглах
            verbosity: 0,
        });

        const pdfDocument = await loadingTask.promise;
        console.log(JSON.stringify(await pdfDocument.getMetadata(), null, 2));
        console.log("outlines ", JSON.stringify(await pdfDocument.getOutline(), null, 2));
        fs.writeFile("test-outlines.json", JSON.stringify(await pdfDocument.getOutline(), null, 2))
        // Extract text from all pages
        const rawDocs = [];
        for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {
            const page = await pdfDocument.getPage(pageNum);
            console.log(`Processing page ${pageNum}/${pdfDocument.numPages} `, JSON.stringify(await page.getStructTree(), null, 2));
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(' ');

            rawDocs.push({
                pageContent: pageText,
                metadata: {
                    loc: { pageNumber: pageNum, source_path: `page:${pageNum}` },
                },
            });
        }
        console.timeEnd("1. Loading PDF");
        // 2. Text-г chunk-үүдэд хуваах
        console.time("2. Splitting documents");
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 400,
        });
        // ✅ FIX: Actually use the text splitter to chunk the documents
        const docs = await textSplitter.splitDocuments(rawDocs);
        console.log(`Split into ${docs.length} chunks (from ${rawDocs.length} pages).`);
        console.timeEnd("2. Splitting documents");
        await fs.writeFile("test-docs.json", JSON.stringify(docs, null, 2))
        // x2 x3

        // 3. Metadata нэмэх
        docs.forEach(doc => {
            doc.metadata.book_title = pdfFileName;
            doc.metadata.source_path = doc.metadata.loc?.source_path || 'unknown';
            // 'loc.pageNumber' байхгүй тохиолдолд алдаа заахаас сэргийлэх
            doc.metadata.page_number = doc.metadata.loc?.pageNumber || 0;
        });

        // 4. Ensure Weaviate collection exists with proper schema
        console.time("3.1. Ensuring Weaviate schema");
        try {
            const collectionExists = await client.collections.exists(indexName);
            if (!collectionExists) {
                console.log(`Creating new Weaviate collection: ${indexName}`);
                await client.collections.create({
                    name: indexName,
                    properties: [
                        {
                            name: 'content',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'The text content of the document chunk'
                        },
                        {
                            name: 'book_title',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'Title of the source PDF'
                        },
                        {
                            name: 'page_number',
                            dataType: 'int', // Fixed: was ['int']
                            description: 'Page number in the PDF'
                        },
                        {
                            name: 'source_path',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'File path of the PDF'
                        }
                    ],
                    vectorizer: 'none' // We provide embeddings manually
                });
                console.log(`✅ Created collection: ${indexName}`);
            } else {
                console.log(`✅ Collection already exists: ${indexName}`);
            }
        } catch (schemaErr) {
            console.error('⚠️ Schema check/create error (will try to continue):', schemaErr.message);
        }
        console.timeEnd("3.1. Ensuring Weaviate schema");

        // 4. Vector DB-д хадгалах
        console.time("3.2. Storing vectors to Weaviate");
        await WeaviateStore.fromDocuments(docs, embeddings, {
            client,
            indexName,
            textKey: 'content',
            metadataKeys: ['book_title', 'page_number', 'source_path'],
        });
        console.timeEnd("3.2. Storing vectors to Weaviate");

        console.log(`✅ PDF '${pdfFileName}' vectors saved to Weaviate under index '${indexName}'`);
        console.timeEnd(`Ingestion process for ${pdfPath}`);
        return { ok: true, message: "Success", pdf: pdfFileName, indexName, docCount: docs.length };

    } catch (err) {
        console.error("❌ Error ingesting PDF:", err.stack || err.message);
        return { ok: false, error: err.message };
    }
}


async function askQuestion(query, indexName, bookName, conversationId, pdfUrl, currentPage, secondPage) {





    let conversationHistory = [];
    if (conversationId && (conversationId + "").length > 0) {
        conversationHistory = await supabase.from("chats").select("*").eq("conversation_id", conversationId).order("created_at", {
            ascending: true
        }).limit(20).then(e => e.data)
    }

    const formattedContext = (conversationHistory || [])
        .map(m => {
            // хэрвээ мессеж рол мэдэгдэхгүй бол асуулт/хариултаар таамаглана
            const q = m.question;
            const a = m.answer;
            return `User: ${q}\nAssistant: ${a}`;
        })
        .join('\n---\n');
    const genAI = new GoogleGenAI(process.env.GOOGLE_API_KEY);


    const pdfResponse = await fetch(pdfUrl);
    if (!pdfResponse.ok) {
        throw new Error('Failed to fetch PDF');
    }
    const arrayBuffer = await pdfResponse.arrayBuffer();
    const base64Data = Buffer.from(arrayBuffer).toString('base64');



    const nodeModulesPath = path.resolve(__dirname, '../../node_modules/pdfjs-dist');
    const cmapsPath = path.join(nodeModulesPath, 'cmaps').replace(/\\/g, '/') + '/';
    const fontsPath = path.join(nodeModulesPath, 'standard_fonts').replace(/\\/g, '/') + '/';

    console.log('✅ PDF.js paths:', { cmapsPath, fontsPath });

    const loadingTask = pdfjs.getDocument({
        data: arrayBuffer,
        cMapUrl: cmapsPath,
        cMapPacked: true,
        standardFontDataUrl: fontsPath,
        useSystemFonts: true, // Систем дэх Япон fontуудыг ашиглах
        verbosity: 0,
    });

    const pdfDocument = await loadingTask.promise;

    let text = "";

    // ✅ Хоёр эсвэл нэг хуудасны текстийг авах
    if (currentPage && !isNaN(currentPage) && currentPage <= pdfDocument.numPages) {
        const page = await pdfDocument.getPage(currentPage);

        text += `==================== page_number:${currentPage} ====================\n`;
        const textContent = await page.getTextContent();
        text += textContent.items.map(item => item.str).join(' ');

        // Хэрэв secondPage байвал түүнийг ч нэмнэ (double page view)
        if (secondPage && !isNaN(secondPage) && secondPage <= pdfDocument.numPages) {
            const nextPage = await pdfDocument.getPage(secondPage);
            text += `\n==================== page_number:${secondPage} ====================\n`;
            const nextTextContent = await nextPage.getTextContent();
            text += nextTextContent.items.map(item => item.str).join(' ');
        } else if (!secondPage) {
            // Single page mode бол дараагийн хуудсыг context-ийн тулд нэмнэ
            let pageEnd = currentPage + 1;
            if (pageEnd <= pdfDocument.numPages) {
                const nextPage = await pdfDocument.getPage(pageEnd);
                text += `\n==================== page_number:${pageEnd} ====================\n`;
                const nextTextContent = await nextPage.getTextContent();
                text += nextTextContent.items.map(item => item.str).join(' ');
            }
        }
    }

    const rawDocs = [];
    // for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {
    //     const page = await pdfDocument.getPage(pageNum);
    //     console.log(`Processing page ${pageNum}/${pdfDocument.numPages} `, JSON.stringify(await page.getStructTree(), null, 2));
    //     const textContent = await page.getTextContent();
    //     const pageText = textContent.items.map(item => item.str).join(' ');
    //     text += `==================== page_number:${pageNum} ====================\n${pageText}\n\n`;
    //     rawDocs.push({
    //         pageContent: pageText,
    //         metadata: {
    //             loc: { pageNumber: pageNum, source_path: `page:${pageNum}` },
    //         },
    //     });
    // }

    const qaSystemPrompt = `

🎓 PDF問題練習 AI教師プロンプト（問題出題モード）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
� 絶対厳守ルール - 違反は許されない 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

あなたはPDF教材の問題練習を手伝うAI教師です。

❌❌❌ 以下は**絶対禁止**です:
1. ページ全体の説明 (例: "このページは〇〇です")
2. 理論・概念の説明 (例: "正負の数とは...")
3. 「何から始めたいですか？」のような質問
4. 「イメージはありますか？」のような問いかけ
5. 長い挨拶 (3行以上)
6. 問題を出さずに終わる返答
7. 「このページでは〇〇について学びます」
8. 「数直線で考えると...」のような概念説明
9. ページ内容の要約やリスト化
10. 「このページの最初の問題に挑戦しましょう！」(問題の式がない)

✅✅✅ 必ずやること:
1. **すぐに具体的な問題を出題**
2. 問題番号を明記 (例: "(1)")
3. 問題の式をそのまま書く (例: "−0.9 + 0.7 =")
4. 最大3行以内
5. 毎回必ず問題を含める

【重要】あなたが返答するたびに、**必ず問題の式 (例: "−0.9 + 0.7 =")** が含まれていなければなりません。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 言語ルール
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

すべての応答は**必ず日本語のみ**で生成してください。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 教材情報
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

生徒からの質問: ${query}
現在のページ内容: ${text}
${formattedContext.length > 0 ? `会話履歴:\n${formattedContext}` : ""}
${!isNaN(currentPage) ? `📄 現在のページ番号: ${secondPage ? `${currentPage}〜${secondPage}ページ` : `${currentPage}ページ`}` : ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 応答フォーマット
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【❌ 悪い例 - 絶対にやってはいけない】

「このページは**正負の数の足し算**について学びます」
→ ❌ 問題の式がない！理論説明！

「こんにちは！📚 一緒に勉強しましょう！

このページの最初の問題に挑戦しましょう！」
→ ❌ 問題の式がない！

【✅ 良い例 - これが正解】

yag「(1) −0.9 + 0.7 = 

やってみて！」
→ ✅ 問題番号 + 式！

「正解！🎉

(2) −1.6 + 0.8 = 」
→ ✅ すぐ次の問題！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 初回メッセージ（会話履歴が空の場合）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 良い例:
「(1) −0.9 + 0.7 = 

やってみて！」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 問題進行ルール - 超重要！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【ステップ1: 会話履歴を確認】
- 前回どの問題を出題したか？ (例: "(1) −0.9 + 0.7 =")
- 生徒が何を答えたか？

【ステップ2: 答えをチェック】
生徒の答えが数字の場合:
1. **PDFページに答え(answer key)が書かれています**
   例: "(1) −0.9 + 0.7 = −0.2" のように答えが併記されています
2. PDFから正解を抽出する（数字の部分: −0.2）
3. 生徒の答え（例: "-0.2"）と比較する
   - **重要**: "−0.2" と "-0.2" は同じ（マイナス記号の違いは無視）
   - 数値として一致するかチェック
4. **完全一致の場合のみ** → 「正解！🎉」と褒めて次の問題(2)を出題
5. 一致しない場合 → 「惜しい！答えは −0.2 です」と正解を教えて同じ問題(1)を再出題

重要な例:
- PDFに "(1) −0.9 + 0.7 = −0.2" と書かれている
- 正解は: −0.2
- 生徒が "-0.2" と答えた → **数値として一致** → ✅ 正解！次の問題(2)へ
- 生徒が "−0.2" と答えた → **数値として一致** → ✅ 正解！次の問題(2)へ
- 生徒が "0.2" と答えた → **数値が違う** → ❌ 不正解！「答えは −0.2 です」と教える
- 生徒が "654" と答えた → **数値が違う** → ❌ 不正解！「答えは −0.2 です」と教える

**チェック方法**: PDFの正解と生徒の答えを数値として比較してください。

【ステップ3: 返答フォーマット】

✅ 正解の場合:
「正解！🎉

(2) −1.6 + 0.8 = 

やってみて！」

❌ 不正解の場合:
「惜しい！答えは [正しい答え] です。

もう一度、同じ問題を解いてみましょう:
[同じ問題の式]」

**重要**: 不正解の場合、必ず**現在出題した問題番号と式**をそのまま再出題してください。
(1)を出題したなら(1)を、(5)を出題したなら(5)を再度出してください。

【ステップ4: 特別なメッセージ】
- "わかりました！自分で解いてみます" → 「頑張って！💪 答えを教えてね\n\n(1) −0.9 + 0.7 = 」(同じ問題の式を再度表示)
- "次の問題" または "スキップ" または "この問題をスキップして" → **必ず次の番号の問題を出題**
  例: 今 (1) なら → (2) を出題
      今 (5) なら → (6) を出題
  同じ問題を繰り返さない！
- "-0.2" のような数字 → 答えをチェック！（ステップ2参照）

**超重要**: 生徒が「スキップ」や「次の問題」と言った場合、同じ問題を繰り返してはいけません。
必ず次の問題番号に進んでください。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 実例フロー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【会話履歴が空】
→ 「(1) −0.9 + 0.7 = \n\nやってみて！」

【生徒: "−0.2"】
→ 正解チェック → ✅ 正しい！
→ 「正解！🎉\n\n(2) −1.6 + 0.8 = 」

【生徒: "654"（問題(1)への答え）】
→ 正解チェック → ❌ 間違い！ (正解: −0.2)
→ 「惜しい！答えは −0.2 です。\n\nもう一度: (1) −0.9 + 0.7 = 」

【生徒: "-0.8"（問題(2)への答え）】
→ 正解チェック → ❌ 間違い！ (正解: −0.8)
→ 「惜しい！答えは −0.8 です。\n\nもう一度: (2) −1.6 + 0.8 = 」
→ **重要**: (2)を出題していたので、(2)を再出題！(1)に戻らない！

【生徒: "次の問題" または "この問題をスキップして次の計算問題を教えてください"】
→ 会話履歴で (1) を出題済み確認
→ **同じ (1) を繰り返さない！**
→ 必ず次へ進む: 「(2) −1.6 + 0.8 = \n\nやってみて！」

【生徒が何度も "スキップ" と言う場合】
→ 毎回次の問題番号に進む
例: (1) → skip → (2) → skip → (3) → skip → (4)
→ **絶対に同じ問題を繰り返さない**

さあ、PDFの問題を順番通りに出題しましょう！📚
        `;

    const contents = [
        {
            role: "model",
            parts: [
                (text && text.length > 0) ? ({
                    type: "text",
                    text: `【PDF教材の内容】\n${text}`
                }) : ({
                    inlineData: {
                        mimeType: 'application/pdf',
                        data: base64Data,
                    },
                }),

            ]
        },
        // ✅ FEW-SHOT EXAMPLES - Gemini-д format заах
        {
            role: "user",
            parts: [{ text: "このページの問題を教えてください", type: "text" }]
        },
        {
            role: "model",
            parts: [{ text: "(1) −0.9 + 0.7 = \n\nやってみて！", type: "text" }]
        },
        {
            role: "user",
            parts: [{ text: "-0.2", type: "text" }]
        },
        {
            role: "model",
            parts: [{ text: "正解！🎉\n\n(2) −1.6 + 0.8 = ", type: "text" }]
        },
        {
            role: "user",
            parts: [{ text: "次の問題", type: "text" }]
        },
        {
            role: "model",
            parts: [{ text: "(3) −0.7 + 0.6 = \n\nやってみて！", type: "text" }]
        },
        // ✅ ACTUAL USER QUESTION
        {
            role: "user",
            parts: [
                {
                    type: "text",
                    text: query
                }
            ]
        }
    ]
    console.log({ text })
    const ai = new GoogleGenAI({
        apiKey: process.env.GOOGLE_API_KEY,
    });
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-lite',
        contents,
        config: {
            temperature: 0.3,  // Lower temperature for consistent answer checking
            topP: 0.95,
            topK: 40,
             systemInstruction: qaSystemPrompt
        },
        generationConfig: {
            temperature: 0.3,  // Lower temperature for consistent answer checking
            topP: 0.95,
            topK: 40,
        },
    });

    const extractedText = response.candidates[0].content.parts[0].text

    // ✅ Extract token usage information from Gemini response
    const usageMetadata = response.usageMetadata || {};
    const tokenUsage = {
        promptTokens: usageMetadata.promptTokenCount || 0,
        candidatesTokens: usageMetadata.candidatesTokenCount || 0,
        totalTokens: usageMetadata.totalTokenCount || 0,
    };

    console.log("📊 Token Usage:", tokenUsage);

    return {
        candidates: response.candidates,
        answer: extractedText,
        question: query,
        tokenUsage: tokenUsage // ✅ Include token usage in response
    }

    //         // ## あなたの役割

    //         // 1. **質問に答えるだけでなく、積極的に教える**
    //         //    - 生徒が集中力を保てるよう、段階的にPDFの内容を案内します
    //         //    - ただ質問を待つのではなく、理解度を確認し、次へ進むよう促します

    //         // 2. **日本語で優しく指導する**
    //         //    - 常に日本語で話します（モンゴル語は使わない）
    //         //    - 難しい漢字や専門用語は、分かりやすく説明します
    //         //    - 褒めて励まし、学習意欲を高めます

    //         // 3. **インタラクティブな学習体験を提供**
    //         //    - 内容を説明した後、「分かりましたか？」と確認します
    //         //    - 理解度チェックのための簡単な質問をします
    //         //    - 漢字の読み方や意味を教えます
    //         //    - 具体例を出して説明します

    //         // ## 指導の流れ

    //         // ### 最初のメッセージ（会話履歴が空の場合）
    //         // もし会話履歴が空っぽなら、このように始めてください：

    //         // 「こんにちは！一緒にこの教材を学びましょう。📚

    //         // 最初のページから始めますね。まず、内容を読んでみましょう。

    //         // [ここで最初のページの重要なポイントを簡潔に説明する]

    //         // この部分は理解できましたか？分からないところがあれば、遠慮なく聞いてくださいね。」

    //         // ### 会話が続いている場合
    //         // - 生徒の質問に答えた後、「他に質問はありますか？」と聞く
    //         // - 理解できたようなら、「よくできました！次のページに進みましょうか？」と促す
    //         // - 難しい言葉があれば、「この漢字『○○』の意味は分かりますか？」と確認する

    //         // ## 重要なルール

    //         // ✅ **必ずすること**
    //         // - 教材の内容に基づいて教える
    //         // - 日本語のみで話す,必要に応じてモンゴル語で指示を出す
    //         // - 褒めて励ます
    //         // - 理解度を確認する質問をする
    //         // - 段階的に進める

    //         // ❌ **してはいけないこと**
    //         // - 教材にない情報を勝手に作らない
    //         // - 一度に多くの情報を詰め込まない
    //         // - 生徒を置いて先へ進まない
    //         // - 冷たい態度や機械的な対応



    //     const client = await makeWeaviateClient();
    //     console.log({
    //         query, indexName, bookName, conversationId
    //     })
    //     let conversationHistory = [

    //     ];

    //     if (conversationId && (conversationId + "").length > 0) {
    //         conversationHistory = await supabase.from("chats").select("*").eq("conversation_id", conversationId).order("created_at", {
    //             ascending: true
    //         }).limit(20).then(e => e.data)
    //     }

    //     try {
    //         console.time("Total question answering time");
    //         console.log(`Querying index '${indexName}' for book '${bookName}'`);
    //         const vectorStore = await WeaviateStore.fromExistingIndex(embeddings, {
    //             client,
    //             indexName: indexName,
    //             textKey: 'content',
    //             metadataKeys: ['book_title', 'page_number', 'source_path'],
    //         });

    //         // LangChain JS-д зориулсан where филтерийг ашиглах
    //         // Энэ нь зөвхөн тухайн номын chunk-үүдээс хайлт хийнэ.
    //         const weaviateFilter = {
    //             operator: "Like",              // "Like" эсвэл "NotLike"

    //             path: ['book_title', "content"],
    //             valueText: query,
    //         };

    //         const retriever = vectorStore.asRetriever({
    //             k: 5,
    //             searchKwargs: {
    //                 where: weaviateFilter // where филтер ашиглах
    //             }
    //         });

    //         //         const qaSystemPrompt = `
    //         // あなたは優しく、忍耐強い日本語の先生です。生徒がこのPDF教材を理解し、一歩一歩学ぶのを手伝います。

    //         // ## あなたの役割

    //         // 1. **質問に答えるだけでなく、積極的に教える**
    //         //    - 生徒が集中力を保てるよう、段階的にPDFの内容を案内します
    //         //    - ただ質問を待つのではなく、理解度を確認し、次へ進むよう促します

    //         // 2. **日本語で優しく指導する**
    //         //    - 常に日本語で話します（モンゴル語は使わない）
    //         //    - 難しい漢字や専門用語は、分かりやすく説明します
    //         //    - 褒めて励まし、学習意欲を高めます

    //         // 3. **インタラクティブな学習体験を提供**
    //         //    - 内容を説明した後、「分かりましたか？」と確認します
    //         //    - 理解度チェックのための簡単な質問をします
    //         //    - 漢字の読み方や意味を教えます
    //         //    - 具体例を出して説明します

    //         // ## 指導の流れ

    //         // ### 最初のメッセージ（会話履歴が空の場合）
    //         // もし会話履歴が空っぽなら、このように始めてください：

    //         // 「こんにちは！一緒にこの教材を学びましょう。📚

    //         // 最初のページから始めますね。まず、内容を読んでみましょう。

    //         // [ここで最初のページの重要なポイントを簡潔に説明する]

    //         // この部分は理解できましたか？分からないところがあれば、遠慮なく聞いてくださいね。」

    //         // ### 会話が続いている場合
    //         // - 生徒の質問に答えた後、「他に質問はありますか？」と聞く
    //         // - 理解できたようなら、「よくできました！次のページに進みましょうか？」と促す
    //         // - 難しい言葉があれば、「この漢字『○○』の意味は分かりますか？」と確認する

    //         // ## 重要なルール

    //         // ✅ **必ずすること**
    //         // - 教材の内容に基づいて教える
    //         // - 日本語のみで話す,必要に応じてモンゴル語で指示を出す
    //         // - 褒めて励ます
    //         // - 理解度を確認する質問をする
    //         // - 段階的に進める

    //         // ❌ **してはいけないこと**
    //         // - 教材にない情報を勝手に作らない
    //         // - 一度に多くの情報を詰め込まない
    //         // - 生徒を置いて先へ進まない
    //         // - 冷たい態度や機械的な対応
    //         // <context>
    //         // {context}
    //         // </context>`;

    //         const qaSystemPrompt = `
    // 
    0;
    //         const { ChatPromptTemplate, MessagesPlaceholder } = require('@langchain/core/prompts');
    //         const { createStuffDocumentsChain } = require('langchain/chains/combine_documents');
    //         const { createRetrievalChain } = require('langchain/chains/retrieval');

    //         const prompt = ChatPromptTemplate.fromMessages([
    //             ['system', qaSystemPrompt],
    //             new MessagesPlaceholder('history'), // 👈 энэ бол өмнөх яриаг оруулах хэсэг

    //             ['human', '{input}'],
    //         ]);

    //         const questionAnswerChain = await createStuffDocumentsChain({ llm, prompt });
    //         const chain = await createRetrievalChain({
    //             retriever,
    //             combineDocsChain: questionAnswerChain,

    //         });

    //         const chatHistory = [];

    //         for (const msg of conversationHistory) {
    //             // Шинэ schema: { message, role: "USER" | "AI" }
    //             if (msg.role === "USER") {
    //                 chatHistory.push({ role: 'user', content: msg.message });
    //             } else if (msg.role === "AI") {
    //                 chatHistory.push({ role: 'assistant', content: msg.message });
    //             }
    //             // Fallback: хуучин schema { question, answer } (backward compatibility)
    //             else {
    //                 if (msg.question) {
    //                     chatHistory.push({ role: 'user', content: msg.question });
    //                 }
    //                 if (msg.answer) {
    //                     chatHistory.push({ role: 'assistant', content: msg.answer });
    //                 }
    //             }
    //         }

    //         console.time("Chain invocation time");
    //         const response = await chain.invoke({ input: query, history: chatHistory });
    //         console.log(chatHistory);
    //         console.timeEnd("Chain invocation time");

    //         console.log('\n--- Хариулт ---');
    //         console.log(response.answer);
    //         console.timeEnd("Total question answering time");
    //         console.log({ conversationId, qaSystemPrompt })



    //         return response;
    //     } catch (err) {
    //         console.error('❌ Асуулга асуухад алдаа гарлаа:', err.stack || err.message);
    //         throw err;
    //     }
}

/**
 * PDF-г Gemini Vision ашиглан vector database-д хадгалах
 * Зураг, диаграмм, хүснэгтийн тайлбар орно
 * 
 * @param {string} pdfPath PDF файлын зам
 * @param {string} indexName Weaviate collection нэр
 * @returns {Promise<Object>} Result object
 */
async function ingestPdfWithVision(pdfPath, indexName = "default_books_index") {
    const client = await makeWeaviateClient();

    try {
        console.time(`[Vision] Ingestion process for ${pdfPath}`);

        // Validate file exists
        await fs.access(pdfPath);
        const pdfFileName = path.basename(pdfPath);
        console.log(`[Vision] Processing PDF with Gemini Vision: ${pdfFileName}`);

        // 1. PDF-г base64 болгох
        console.time("[Vision] 1. Reading PDF to base64");
        const pdfBuffer = await fs.readFile(pdfPath);
        const base64Data = pdfBuffer.toString('base64');
        console.timeEnd("[Vision] 1. Reading PDF to base64");

        // 2. Gemini Vision ашиглан PDF агуулга задлах
        console.time("[Vision] 2. Gemini Vision extraction");
        const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
        const model = genAI.getGenerativeModel({
            model: 'gemini-2.5-flash-lite',
            generationConfig: {
                temperature: 0.2,
                topP: 0.95,
                topK: 40,
                maxOutputTokens: 8192, // Increase for larger PDFs
            },
        });

        // 3. PDF агуулга + зургийн тайлбар авах
        const result = await model.generateContent([
            {
                text: `このPDFの内容を完全に抽出してください：

【抽出する情報】
1. ✅ すべてのテキスト内容
2. ✅ 画像の詳細な説明 (図、グラフ、イラスト)
3. ✅ 表の内容 (すべてのセルを含む)
4. ✅ 数式の説明
5. ✅ ページ番号とセクション構造

【出力フォーマット】
各ページを以下の形式で出力:

━━━━━━━━━━━━━━━━━━━━
📄 ページ [番号]
━━━━━━━━━━━━━━━━━━━━

[TEXT]
すべてのテキスト内容をそのまま

[IMAGE]
画像の詳細な説明
• 何が描かれているか
• 色、形、配置
• 重要なポイント

[TABLE]
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| データ | データ | データ |

[FORMULA]
数式: 2x + 3 = 7
説明: xを求める方程式

━━━━━━━━━━━━━━━━━━━━

このフォーマットで、PDFのすべてのページを処理してください。`
            },
            {
                inlineData: {
                    mimeType: 'application/pdf',
                    data: base64Data,
                },
            },
        ]);

        const extractedText = result.response.text();
        console.log(`[Vision] Extracted text length: ${extractedText.length} characters`);
        console.timeEnd("[Vision] 2. Gemini Vision extraction");

        // 4. Text splitter (chunk хийх)
        console.time("[Vision] 3. Text splitting");
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 150,
            separators: ['\n━━━━━━━━━━━━━━━━━━━━\n', '\n\n', '\n', ' ', ''],
        });

        const docs = await textSplitter.createDocuments([extractedText]);
        console.log(`[Vision] Split into ${docs.length} document chunks`);
        console.timeEnd("[Vision] 3. Text splitting");

        // 5. Metadata нэмэх
        docs.forEach((doc, index) => {
            doc.metadata.book_title = pdfFileName;
            doc.metadata.source_path = pdfPath;
            doc.metadata.chunk_index = index;
            doc.metadata.extraction_method = 'gemini_vision';
            doc.metadata.has_images = extractedText.includes('[IMAGE]');
            doc.metadata.has_tables = extractedText.includes('[TABLE]');
            doc.metadata.has_formulas = extractedText.includes('[FORMULA]');
        });

        // 6. Ensure Weaviate collection exists
        console.time("[Vision] 4. Ensuring Weaviate schema");
        try {
            const collectionExists = await client.collections.exists(indexName);
            if (!collectionExists) {
                console.log(`[Vision] Creating new Weaviate collection: ${indexName}`);
                await client.collections.create({
                    name: indexName,
                    properties: [
                        {
                            name: 'content',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'The text content including image descriptions'
                        },
                        {
                            name: 'book_title',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'Title of the source PDF'
                        },
                        {
                            name: 'source_path',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'File path of the PDF'
                        },
                        {
                            name: 'chunk_index',
                            dataType: 'int', // Fixed: was ['int']
                            description: 'Index of this chunk in the document'
                        },
                        {
                            name: 'extraction_method',
                            dataType: 'text', // Fixed: was ['text']
                            description: 'Method used to extract content (gemini_vision or text_only)'
                        },
                        {
                            name: 'has_images',
                            dataType: 'boolean', // Fixed: was ['boolean']
                            description: 'Whether this chunk contains image descriptions'
                        },
                        {
                            name: 'has_tables',
                            dataType: 'boolean', // Fixed: was ['boolean']
                            description: 'Whether this chunk contains table data'
                        },
                        {
                            name: 'has_formulas',
                            dataType: 'boolean', // Fixed: was ['boolean']
                            description: 'Whether this chunk contains mathematical formulas'
                        }
                    ],
                    vectorizer: 'none' // We provide embeddings manually
                });
                console.log(`[Vision] ✅ Created collection: ${indexName}`);
            } else {
                console.log(`[Vision] ✅ Collection already exists: ${indexName}`);
            }
        } catch (schemaErr) {
            console.error('[Vision] ⚠️ Schema check/create error (will try to continue):', schemaErr.message);
        }
        console.timeEnd("[Vision] 4. Ensuring Weaviate schema");

        // 7. Weaviate-д vector embeddings хадгалах
        console.time("[Vision] 5. Storing vectors to Weaviate");
        await WeaviateStore.fromDocuments(docs, embeddings, {
            client,
            indexName,
            textKey: 'content',
            metadataKeys: ['book_title', 'source_path', 'chunk_index', 'extraction_method', 'has_images', 'has_tables', 'has_formulas'],
        });
        console.timeEnd("[Vision] 5. Storing vectors to Weaviate");

        console.log(`[Vision] ✅ PDF '${pdfFileName}' with images/tables saved to Weaviate under index '${indexName}'`);
        console.timeEnd(`[Vision] Ingestion process for ${pdfPath}`);

        return {
            ok: true,
            message: "Success with Vision",
            pdf: pdfFileName,
            indexName,
            docCount: docs.length,
            hasImages: extractedText.includes('[IMAGE]'),
            hasTables: extractedText.includes('[TABLE]'),
            hasFormulas: extractedText.includes('[FORMULA]'),
        };

    } catch (err) {
        console.error("[Vision] ❌ Error ingesting PDF with Vision:", err.stack || err.message);
        return { ok: false, error: err.message };
    }
}

module.exports = { ingestPdfToVectorDB, askQuestion, ingestPdfWithVision }
