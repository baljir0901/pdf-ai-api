"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ingestPdfToVectorDB = ingestPdfToVectorDB;
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
const { GoogleGenAI } = require('@google/genai');
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
    }
    else {
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
    const client = await makeWeaviateClient();
    try {
        console.time(`Ingestion process for ${pdfPath}`);
        await fs.access(pdfPath);
        const pdfFileName = path.basename(pdfPath);
        console.log(`Processing PDF: ${pdfFileName}`);
        // 1. PDF-г унших (Япон хэл дэмжсэн тохиргоо)
        console.time("1. Loading PDF");
        const dataBuffer = await fs.readFile(pdfPath);
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
        fs.writeFile("test-outlines.json", JSON.stringify(await pdfDocument.getOutline(), null, 2));
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
        const docs = rawDocs;
        console.log(`Split into ${docs.length} documents.`);
        console.timeEnd("2. Splitting documents");
        await fs.writeFile("test-docs.json", JSON.stringify(docs, null, 2));
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
            }
            else {
                console.log(`✅ Collection already exists: ${indexName}`);
            }
        }
        catch (schemaErr) {
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
    }
    catch (err) {
        console.error("❌ Error ingesting PDF:", err.stack || err.message);
        return { ok: false, error: err.message };
    }
}
async function askQuestion(query, indexName, bookName, conversationId, pdfUrl, currentPage) {
    let conversationHistory = [];
    if (conversationId && (conversationId + "").length > 0) {
        conversationHistory = await supabase.from("chats").select("*").eq("conversation_id", conversationId).order("created_at", {
            ascending: true
        }).limit(20).then(e => e.data);
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
    if (currentPage && !isNaN(currentPage) && currentPage < (pdfDocument).numPages) {
        const page = await pdfDocument.getPage(currentPage);
        text += `==================== page_number:${currentPage} ====================\n`;
        const textContent = await page.getTextContent();
        text += textContent.items.map(item => item.str).join(' ');
        let pageEnd = currentPage + 1;
        if (pageEnd < pdfDocument.numPages) {
            const nextPage = await pdfDocument.getPage(pageEnd);
            text += `\n==================== page_number:${pageEnd} ====================\n`;
            const nextTextContent = await nextPage.getTextContent();
            text += nextTextContent.items.map(item => item.str).join(' ');
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

🎓 PDF教材 対話型AI数学教師プロンプト
あなたは日本語で数学を教える、優しく忍耐強い対話型の教師です。
あなたの役割は、情報を要約することではなく、生徒と対話するパートナーとして、一歩一歩学習を導くことです。
目的: 生徒と対話し、質問を投げかけることを通じて、生徒が自ら考え、学ぶ手助けをすること。
🌐 出力言語に関する最重要ルール
あなたの応答は、必ず日本語のみで生成してください。他の言語を一切使用してはいけません。
🔍 生徒と教材の情報
生徒からの質問: ${query}
教材の内容: ${text}
${formattedContext.length > 0 ? `**chat_history**:${formattedContext}` : ""}

${!isNaN(currentPage) ? -`**現在のページ:** ${currentPage}` : ""}
この情報に基づき、生徒との対話を開始してください。
🧑‍🏫 AI教師の役割：対話を始めること
要約せず、質問から始める: 教材の内容をリストアップしたり説明したりするのではなく、最初のキーワードを提示して、それについて生徒に質問することから始めます。
一度に一つのことだけを教える: 一回の返信で扱うトピックや用語、公式は一つだけに絞ります。生徒がそれを理解したら、次に進みます。
生徒の答えを待ってから、対話を進める: 質問を投げかけた後は、必ず生徒の返事を待ち、その内容に応じて次の会話を展開します。
📝 指導の流れ
1️⃣ 最初の会話（会話履歴が空の場合）
こんにちは！一緒に数学を学びましょう。📚
早速ですが、この教材の最初の部分を見てみましょう。
[ここで${text}から最初の重要なキーワードや概念を一つだけ取り上げ、詳しい説明はせずに提示します]
ここに 「二次関数」 という言葉が出てきましたね。
この言葉について、何か知っていることはありますか？ 難しく考えずに、思いついたことを教えてください！
2️⃣ 会話が続いている場合
生徒が答えた後:
「いいですね！その調子です。」
「なるほど、面白い視点ですね！」
理解を深める:
「実は、数学でいう「関数」はもう少し違う意味で使われるんです。一緒に見てみましょうか？」
次のステップへ:
「「二次関数」の基本的な意味が分かったので、次はグラフがどんな形になるか見てみましょうか？」
⚠️ 絶対的なルール
✅ 必ず守ること
会話は常に質問で始め、質問で締めくくる。
一度に教えるトピックは一つだけにする。
生徒の返答に基づいて対話を展開する。
常に優しく、励ますような口調を保つ。
❌ 絶対にしてはいけないこと
教材内容の要約やリスト化は絶対にしない。（例：「このページには5つのポイントがあります…」のような説明は厳禁です）
一度に多くの情報を教えようとしない
生徒の返事を待たずに、一方的に話を進めない。
専門用語を説明なしで使わない
        `;
    const contents = [
        {
            role: "model",
            parts: [
                {
                    text: qaSystemPrompt,
                    type: "text"
                },
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
        {
            role: "user",
            parts: [
                {
                    type: "text",
                    text: `上記のPDF教材の内容に基づいて、次の質問に答えてください：${query}`
                }
            ]
        }
    ];
    console.log({ text });
    const ai = new GoogleGenAI({
        apiKey: process.env.GOOGLE_API_KEY,
    });
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-lite',
        contents,
        config: {
            temperature: 0.7,
            topP: 0.95,
            topK: 40,
        },
        generationConfig: {
            temperature: 0.7,
            topP: 0.95,
            topK: 40,
        },
    });
    const extractedText = response.candidates[0].content.parts[0].text;
    return {
        candidates: response.candidates,
        answer: extractedText,
        question: query
    };
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
    // AI Teacher — Сургалтын багшийн дүрэм
    // Үүрэг (Purpose)
    // Чи бол тэвчээртэй, мэдлэгтэй, ойлгомжтой AI багш. Зорилго — сурагчдад тухайн хичээлийн агуулгад тулгуурласан, логик дараалалтай, товч боловч бүрэн хариулт өгөх.
    // Хариулах зарчим
    // Зөвхөн өгөгдсөн хичээлийн материал болон өмнөх харилцан яриаг (chat history) ашиглан хариул.
    // Хэрэв асуултын хариулт хичээлийн материалд олдохгүй бол яг тодорхой тийм хэлээр хэл: “Энэ мэдээлэл хичээлд олдсонгүй.”
    // Дараа нь тухайн сэдвийн хүрээнд ойлгомжтой тайлбар, жишээ болон хэрэгтэй зөвлөмж өгөх.
    // Худал зүйл зохиож болохгүй. Баримтгүй таамаг, ташаа мэдээлэл бүү нэм.
    // Хариултын бүтэц (хамгийн эхэндээс дараалалтай)
    // Товч хариулт: Гол санааг нэг хоёр мөрт багтааж өг.
    // Тайлбар / Жишээ: Алхам дараалал, богино жишээг ашигла.
    // Дүгнэлт / Дараагийн алхам: Давтаж, сурагчид юуг хийхийг санал болго.
    // Мэндчилгээ ба хэлний хэв маяг
    // Хэллэг: Эелдэг, ойлгомжтой, урам өгсөн.
    // Хэрвээ өмнөх харилцан яриа байгаа бол мэндчилгээг давт битгий хэл.
    // Хүссэн тохиолдолд emoji ашиглаж болно (гэхдээ зохимжгүй их бүү хэрэглэ).
    // Баталгаажуулалт болон сурагчийг шалгах асуултууд
    // Хариуныхаа төгсгөлд сурагчийг ойлголтоо шалгах эсвэл бэлдэхэд туслах жижиг асуулту оруул:
    // “Энгийнээр хэлбэл — ?”
    // “Өөрөөр тайлбарлавал — ?”
    // “Энд асуулт байна уу?”
    // Техникийн тэмдэглэл
    // Бүх хариултанд үнэн зөв, шалгагдсан эх сурвалж байхгүй бол таамаг бүү оруул.
    // <context>
    // {context}
    // </context>`;
    //         // - モンゴル語や英語で話さない
    //         console.log("conversationHistory", JSON.stringify(conversationHistory))
    //         const formattedContext = (conversationHistory || [])
    //             .map(m => {
    //                 // хэрвээ мессеж рол мэдэгдэхгүй бол асуулт/хариултаар таамаглана
    //                 const q = m.question;
    //                 const a = m.answer;
    //                 return `User: ${q}\nAssistant: ${a}`;
    //             })
    //             .join('\n---\n');
    //         const MAX_CONTEXT_CHARS = 20_000;
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
            }
            else {
                console.log(`[Vision] ✅ Collection already exists: ${indexName}`);
            }
        }
        catch (schemaErr) {
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
    }
    catch (err) {
        console.error("[Vision] ❌ Error ingesting PDF with Vision:", err.stack || err.message);
        return { ok: false, error: err.message };
    }
}
module.exports = { ingestPdfToVectorDB, askQuestion, ingestPdfWithVision };
