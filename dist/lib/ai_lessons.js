"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ai_lesson_generate = ai_lesson_generate;
const genai_1 = require("@google/genai");
const promises_1 = __importDefault(require("fs/promises")); // promise-based fs
const path_1 = __importDefault(require("path"));
const slugify_1 = __importDefault(require("slugify"));
const wanakana_1 = __importDefault(require("wanakana"));
const pdf_to_img_1 = require("pdf-to-img");
const utils_1 = require("../utils");
const pdf_1 = require("./pdf");
const ai = new genai_1.GoogleGenAI({
    apiKey: process.env.GOOGLE_API_KEY,
});
async function ai_lesson_generate(pathFile, supabase, fileUrl) {
    const user = await supabase.auth.getUser();
    const fileBuffer = await promises_1.default.readFile(pathFile);
    const base64PDF = fileBuffer.toString('base64');
    console.log("[INFO] Collecting streamed response from AI...");
    const contents = [
        {
            role: 'user',
            parts: [
                {
                    inlineData: {
                        data: base64PDF,
                        mimeType: 'application/pdf',
                    },
                },
                {
                    text: await promises_1.default.readFile(path_1.default.join(process.cwd(), "./prompt.txt")).then(e => e.toString()),
                },
            ],
        },
    ];
    console.log("working stream ....");
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-lite',
        config: {
            temperature: 2,
            responseMimeType: "application/json",
        },
        contents,
    });
    const generate_data = JSON.parse(response.candidates?.[0]?.content?.parts?.[0]?.text || '{}'); //JSON.parse(resultText);
    const document = await (0, pdf_to_img_1.pdf)(pathFile, { scale: 1 });
    const file = pathFile;
    const outputDir = path_1.default.dirname(file);
    const imagePath = path_1.default.basename(file, path_1.default.extname(file));
    const options = {
        format: "png",
        out_dir: outputDir,
        out_prefix: imagePath,
        page: 1, // 👈 зөвхөн эхний хуудас
    };
    console.log("pdfPoppler-convert", "pdfPoppler");
    console.log({ file, outputDir, imagePath });
    let imageFile = await document.getPage(1);
    const uploadServerPath = `posters/${Date.now()}/${imagePath}.png`;
    const { error: imageError } = await supabase.storage
        .from("ai_content")
        .upload(uploadServerPath, imageFile);
    const { data: { publicUrl: imagePublicUrl }, } = supabase.storage.from("ai_content").getPublicUrl(uploadServerPath);
    // supabase ai_lessons үүсгэх код 
    const romaji = wanakana_1.default.toRomaji(generate_data.title);
    // slug болгоно
    const slug = (0, slugify_1.default)(romaji, {
        lower: true,
        strict: true // зөвхөн a-z, 0-9, - үүсгэнэ
    }) + "" + (0, utils_1.random1000to9999)();
    console.log(romaji);
    const weavite_collection_id = (0, utils_1.formatWeaviateName)(slug + "-PDF");
    const insert_ai_lesson = await supabase.from("ai_lessons").insert({
        title: generate_data.title,
        category_name: generate_data.category_name,
        cover_image: imagePublicUrl,
        mindmap: generate_data.mindmap,
        type: "PDF",
        is_public: false,
        status: "LOADING",
        ai_meta_config: {
            ai_model_name: "gemini",
        },
        slug: slug,
        weavite_collection_id: weavite_collection_id,
        // @ts-ignore
        owner_id: user.data.user.id,
        ref_url: fileUrl
        // @ts-ignore
    }).select().then(e => e?.data[0]);
    console.log(await supabase.from("ai_summaries").insert(generate_data.summary.map((t) => ({
        title: t.title,
        description: t.description,
        img_desc_positive: t.img_desc.positive,
        img_desc_negative: t.img_desc.negative,
        key_vocabulary: t.key_vocabulary,
        activity_suggestion: t.activity_suggestion,
        quizzes: t.quizzes,
        ai_lessons_id: insert_ai_lesson.id
    }))));
    await (0, pdf_1.ingestPdfToVectorDB)(pathFile, weavite_collection_id);
    console.log("flash card generate start");
    const flashCardResponse = await ai.models.generateContent({
        model: 'gemini-2.5-flash-lite',
        contents: [
            {
                role: 'user',
                parts: [
                    {
                        inlineData: {
                            data: base64PDF,
                            mimeType: 'application/pdf',
                        },
                    },
                    {
                        text: `
    Чи миний PDF-ын текстийг уншиж, flashcard болгон хувиргах үүрэгтэй. 
    Би PDF-ын текстийг доор оруулна. Чи:

    1. Текстийг хэсэгчилж унш.
    2. Гол санааг олж, **асуулт ба хариулт** (Q&A) форматаар гарга.
    3. Summary items:${generate_data.summary.map((e) => e.title).join(",")}
    3. Descriptions: Тухайн сэдэвтэй зориулагдсан тайлбар мэдээлэл
    4. Flashcard-ууд богино, ойлгомжтой, хураангуй байх ёстой.
    5. Flashcard-уудыг дараах JSON форматаар гарга:
    6. Япон хэл дээр мэдээлэл гаграна уу
    [
        {
            summary_title:"Summary title",
            description:"Summary description",
            items:[{
                "question": "Энд асуулт",
                "answer": "Энд хариулт",
                description:"Тухайн flashcard дээрх тайлбар",
            }],
        }
      ...
    ]

    **Жишээ:**
    Text: "Photosynthesis is the process by which plants use sunlight to synthesize foods from carbon dioxide and water."
    Output:
    [
      {
        summary_title:"Photosynthesis",
        description:"lorem ipsum dolar sit amet",
        items:[
            {
                description:"lorem ipsum dolar sit amet",
                "question": "What is photosynthesis?",
                "answer": "The process by which plants use sunlight to synthesize foods from carbon dioxide and water."
            }
        ]
      }
    ]                    `,
                    },
                ],
            }
        ],
        config: {
            temperature: 2,
            responseMimeType: "application/json",
        },
    });
    console.log("generate flash cards");
    //     fs.writeFile("flashcards.json", flashCardResponse.candidates[0].content?.parts[0].text, { encoding: "utf8" })
    const generate_flashCards = JSON.parse(flashCardResponse.candidates?.[0]?.content?.parts?.[0]?.text || '[]');
    console.log("flash card insert db");
    const ai_flash_cards_insert_values = await supabase.from("ai_flash_cards").insert(generate_flashCards.map((e) => ({
        title: e.summary_title,
        description: e.description,
        ai_lesson_id: insert_ai_lesson.id
    }))).select("*").then(e => {
        return e.data;
    });
    const ai_flash_card_items = ai_flash_cards_insert_values?.reduce((pre, curr, index) => {
        return [
            ...pre,
            ...generate_flashCards[index].items.map((t) => ({
                flash_card_id: curr.id,
                title: t.question,
                description: t.answer
            }))
        ];
    }, []);
    console.log(JSON.stringify(ai_flash_card_items));
    await supabase.from("ai_flash_card_items").insert(ai_flash_card_items);
    const defaultConversations = await ai.models.generateContent({
        model: 'gemini-2.5-flash-lite',
        contents: [
            {
                role: 'user',
                parts: [
                    {
                        inlineData: {
                            data: base64PDF,
                            mimeType: 'application/pdf',
                        },
                    },
                    {
                        text: `
   Чи бол боловсролын chatbot-т туслах туслах. 

Би чамд нэг PDF хичээлийн текст өгнө. 
Чиний үүрэг: энэ хичээлд тулгуурласан **default conversation list** гаргах. 

Шаардлага:
1. Хариу **JSON array** хэлбэртэй байх. 
2. Бүх item-д дараах талбарууд байх:
   - "id": 1,2,3,...  
   - "title": Хичээлийн бүлэг, гарчиг, эсвэл main topic
   - "prompt": Хэрэглэгч асууж болох асуулт
   - "difficulty": "easy", "medium", эсвэл "hard"
   - "category": Хичээлийн төрөл, жишээ: "theory", "exercise", "example"
   - "hint": Хэрэглэгчдэд өгөх жижиг зөвлөмж / заавар (optional)
3. Хариу нь зөвхөн JSON байх, тайлбар, markdown, текст нэмэхгүй.
4. JSON нь зөв бүтэцтэй байх ёстой, бүх талбарууд заавал байх.

Жишээ JSON формат:

[
  { 
    "id": 1,
    "title": "Сэдэв 1: Танилцуулга",
    "prompt": "Энэ хичээлийн гол сэдвийг тайлбарла?",
    "difficulty": "easy",
    "category": "theory",
    "hint": "Хичээлийн эхний paragraph-ийг унш"
  },
  { 
    "id": 2,
    "title": "Сэдэв 2: Дадлага",
    "prompt": "Жишээ даалгаврыг шийд",
    "difficulty": "medium",
    "category": "exercise",
    "hint": "Дадлагын хэсгийг дахин унш"
  }
]     `,
                    },
                ],
            }
        ],
        config: {
            temperature: 2,
            responseMimeType: "application/json",
        },
    });
    await supabase.from("ai_lessons").update({
        status: "FINISH",
    }).eq("id", insert_ai_lesson.id);
    console.log("FINISH WORKING");
    const default_conversation_values = (JSON.parse(defaultConversations.candidates?.[0]?.content?.parts?.[0]?.text || '[]'));
    console.log(await supabase.from("ai_lesson_default_conversations").insert(default_conversation_values.map((c) => {
        return {
            ai_lessons_id: insert_ai_lesson.id,
            text: c.title,
            type: c.difficulty,
            prompt: c.prompt,
            hint: c.hint,
        };
    })));
    return insert_ai_lesson;
}
const flashCards = [
    {
        "summary_title": "プログラミング言語って何？",
        "description": "コンピュータと人間がコミュニケーションをとるための「言葉」、プログラミング言語の基本的な概念と、その発展の歴史を学びます。",
        "items": [
            {
                "question": "プログラミング言語とは何ですか？",
                "answer": "人間とコンピュータの間の「橋」の役割を果たし、プログラムのコマンドを人間と機械の両方が理解できる形で表現するためのツールです。",
                "description": "人間が考える「やりたいこと」をコンピュータが実行できる命令に翻訳するための言葉とルールセットです。"
            },
            {
                "question": "プログラミング言語の歴史は、どのような言語から始まりましたか？",
                "answer": "初期のコンピュータでは手作業で命令が入力されていましたが、その後、機械語に近い「アセンブリ言語」が誕生しました。これがプログラミング言語の始まりです。",
                "description": "アセンブリ言語は低水準言語と呼ばれ、コンピュータのハードウェアに非常に近いレベルの操作を記述します。"
            },
            {
                "question": "高水準言語にはどのようなものがありますか？",
                "answer": "人間が理解しやすいように設計された言語で、FORTRAN、COBOL、C、Pascal、そしてJavaや.NETなどが含まれます。",
                "description": "高水準言語は、より抽象的で、特定のコンピュータの構造に依存しないため、開発効率が向上します。"
            }
        ]
    },
    {
        "summary_title": "Javaの誕生と歴史",
        "description": "Javaがどのようにして生まれ、なぜこれほどまでに広く使われるようになったのか、その誕生の背景と歴史的な経緯を解説します。",
        "items": [
            {
                "question": "Javaはいつ、誰によって開発が始まりましたか？",
                "answer": "1991年にサン・マイクロシステムズ社（Sun Microsystems）のジェームズ・ゴスリン氏を中心とするチームによって開発が始まりました。",
                "description": "当初は家電製品向けのプラットフォームとして開発されていました。"
            },
            {
                "question": "Javaの最初の名前は何でしたか？",
                "answer": "「Oak」（オーク）という名前でした。しかし、商標の問題で後に「Java」に変更されました。",
                "description": "Javaという名前の由来は、開発者がよく飲んでいたコーヒーの種類から来ているという説が有名です。"
            },
            {
                "question": "Javaが広く普及したきっかけは何ですか？",
                "answer": "1990年代のWorld Wide Web（WWW）の急速な普及です。プラットフォームに依存しないJavaの特性が、Webアプリケーション開発に非常に適していました。",
                "description": "OSやCPUに依存せずにプログラムを動かせるというJavaの強みが、インターネット時代にマッチしました。"
            }
        ]
    },
    {
        "summary_title": "Javaのすごいところ！",
        "description": "Javaが他の多くのプログラミング言語と比べて優れている点、特に「プラットフォーム非依存性」という最大の特徴について学びます。",
        "items": [
            {
                "question": "Javaの最大の特徴である「プラットフォーム非依存性」とは何ですか？",
                "answer": "「一度書けば、どこでも実行できる（Write Once, Run Anywhere）」というスローガンで知られる、OS（Windows, Mac, Linuxなど）を問わずにプログラムを動作させられる性質のことです。",
                "description": "この特徴により、開発者はOSごとにプログラムを書き直す必要がなくなります。"
            },
            {
                "question": "Javaはどのようにしてプラットフォーム非依存性を実現していますか？",
                "answer": "Java仮想マシン（JVM）という仕組みを利用しています。Javaのコードは特定のOS用ではなく、JVM用の「バイトコード」に変換され、各OSにインストールされたJVMがそれを解釈して実行します。",
                "description": "ソースコード → (コンパイラ) → バイトコード → (JVM) → 実行、という流れです。"
            },
            {
                "question": "初心者がJavaを学ぶメリットは何ですか？",
                "answer": "比較的文法がわかりやすく、プログラミングの基本概念をしっかり学べます。また、オブジェクト指向の考え方が完全に実装されているため、他の言語を学ぶ際の強固な基礎となります。",
                "description": "世界中で多くの開発者に利用されており、学習資料やコミュニティが豊富な点もメリットです。"
            }
        ]
    },
    {
        "summary_title": "プログラミングの準備をしよう",
        "description": "実際にJavaプログラミングを始めるための環境構築を行います。JDKのインストールから、コマンドラインでJavaを使えるようにする設定までを解説します。",
        "items": [
            {
                "question": "Javaで開発を始めるために必要なものは何ですか？",
                "answer": "JDK（Java Development Kit）です。これには、プログラムをコンパイル（変換）するツールや実行環境（JRE）など、開発に必要なものがすべて含まれています。",
                "description": "JRE (Java Runtime Environment)はプログラムを実行するだけの環境ですが、JDKは開発もできる環境です。"
            },
            {
                "question": "JDKをインストールした後に行う「環境変数の設定」とは何ですか？",
                "answer": "コンピュータのどのフォルダからでもJavaのコンパイル（javac）や実行（java）コマンドを呼び出せるように、コマンドの場所をOSに登録する作業です。",
                "description": "具体的には、環境変数の`PATH`にJDK内の`bin`フォルダのパスを追加します。"
            },
            {
                "question": "環境設定が正しくできたか確認するには、どのコマンドを使いますか？",
                "answer": "コマンドプロンプト（またはターミナル）を開き、「javac」と入力してEnterキーを押します。オプションの一覧が表示されれば設定は成功です。",
                "description": "「'javac' is not recognized...」のようなエラーが出た場合は、PATHの設定が間違っている可能性が高いです。"
            }
        ]
    },
    {
        "summary_title": "はじめてのJavaプログラム",
        "description": "簡単な「Hello World」プログラムを例に、Javaプログラムの作成、コンパイル、実行までの一連の流れを体験します。",
        "items": [
            {
                "question": "Javaのソースファイルを作成する際の最も重要なルールは何ですか？",
                "answer": "publicなクラスのクラス名と、ファイル名を完全に一致させる必要があります。例えば、「public class MyFirstProgram」なら、ファイル名は「MyFirstProgram.java」とします。",
                "description": "大文字と小文字も区別されるため、正確に一致させる必要があります。"
            },
            {
                "question": "Javaプログラムの実行は、どの部分から始まりますか？",
                "answer": "「public static void main(String[] args)」という特別なメソッド（mainメソッド）から実行が開始されます。プログラムのエントリーポイント（入口）です。",
                "description": "どんなに大きなプログラムでも、必ずこのmainメソッドが最初の実行点となります。"
            },
            {
                "question": "Javaソースコードをコンパイル（変換）するコマンドは何ですか？",
                "answer": "「javac ファイル名.java」です。このコマンドを実行すると、JVMが読み取れるバイトコードが書かれた「.class」ファイルが生成されます。",
                "description": "例: `javac MyFirstProgram.java` を実行すると `MyFirstProgram.class` が作られます。"
            },
            {
                "question": "コンパイルされたプログラムを実行するコマンドは何ですか？",
                "answer": "「java クラス名」です。このとき、ファイル名の「.class」は付けません。",
                "description": "例: `java MyFirstProgram` を実行すると、プログラムが起動します。"
            }
        ]
    },
    {
        "summary_title": "もっと挑戦してみよう！",
        "description": "基本的なプログラムの理解を深めるための、いくつかの応用的な課題について学びます。",
        "items": [
            {
                "question": "文字を出力する際の `print()` と `println()` の違いは何ですか？",
                "answer": "`println()` は指定した文字を出力した後に改行しますが、`print()` は改行しません。",
                "description": "`println`は「print line」の略だと覚えると分かりやすいです。"
            },
            {
                "question": "`main`メソッドに引数を渡すにはどうすればよいですか？",
                "answer": "プログラムを実行する際に `java クラス名 引数1 引数2 ...` のようにコマンドラインで指定します。渡された引数は `String[] args` という配列に格納されます。",
                "description": "プログラムの動作を外部からコントロールしたい場合に便利な機能です。"
            },
            {
                "question": "1つのクラスに `main` メソッドを2つ書くことはできますか？",
                "answer": "できません。Javaでは、メソッドは名前と引数の型の組み合わせで区別されます。「public static void main(String[] args)」というシグネチャ（署名）を持つメソッドは、1つのクラスに1つしか定義できません。",
                "description": "引数の型を変えれば同名のメソッドを複数定義（オーバーロード）できますが、エントリーポイントとなる`main`メソッドは1つだけです。"
            }
        ]
    }
];
const data = {
    "category_name": "プログラミング",
    "video_prompt": "このビデオは、10歳から15歳の子供たちに向けた、5分から10分のアニメ形式の教育ストーリーです。Javaプログラミングの基本概念を、楽しく想像力豊かに解説します。物語は、ジャバボットという賢くてフレンドリーなロボットが、子供たちをプログラミングの世界へ案内するところから始まります。子供たちはジャバボットと一緒にタイムスリップし、FORTRANやC言語のような古いプログラミング言語がどのように使われていたかを見学します。その後、現代に戻り、Javaが「一度書けば、どこでも動く」という魔法のような哲学でなぜ特別なのかを学びます。ジャバボットは、画面にホログラムでコードを映し出しながら、最初のプログラム「こんにちは、世界！」の書き方をステップバイステップで教えます。クラス、メソッド、そしてJava仮想マシン（JVM）という少し難しい概念も、カラフルなアニメーションと分かりやすい比喩で、まるで面白いパズルを解くように説明されます。ビジュアルは、明るい教室、魔法のような風景、そして新しいことを学ぶ喜びにあふれたキャラクターで構成されています。アニメーションのスタイルは日本のものにインスパイアされており、表現豊かな顔の表情、滑らかな動き、鮮やかな光の演出が特徴です。各シーンは、子供たちが重要なレッスンを遊び心あふれる、感情的にポジティブなトーンで理解できるよう作られています。ナレーターは温かく親しみやすい声で、子供たちをワクワクする発見の旅へと導きます。BGMは優しく、学習への集中力と好奇心を高めます。実在の人物、ブランドロゴ、特定の製品への言及は一切なく、すべてがオリジナルで架空のアニメキャラクターで描かれます。除外要素: 醜い、ぼやけている、低解像度、テキスト、透かし、不適切な人体構造、変形。コンテンツは日本語です。",
    "title": "Javaプログラミングへの第一歩",
    "main_poster": {
        "positive": "A friendly anime robot character, 'Javabot,' holding a steaming coffee cup, teaching curious kids Java programming using a holographic screen in a futuristic classroom, vibrant colors, high detail.",
        "negative": "ugly, blurry, low-resolution, text, watermark, bad anatomy, deformed"
    },
    "description": "コンピューターと話すための言葉「プログラミング言語」の不思議な世界へようこそ！このレッスンでは、まずプログラミングがどんなものか、そして昔のコンピューターから現代のインターネットまで、どのように進化してきたかの歴史を探検します。主役は、世界中で大人気のプログラミング言語「Java」です。Javaがなぜ多くの人に選ばれるのか、その秘密である「一度書けば、どこでも動く」という魔法の仕組みを解き明かします。さらに、自分のパソコンでJavaプログラミングを始めるための準備から、実際に「こんにちは！」と表示させる最初のプログラムの作成、そしてそのプログラムがどんな意味を持つのかまで、一つひとつ丁寧に解説します。この冒険が終わる頃には、あなたもプログラマーとしての第一歩を踏み出しているはずです！",
    "summary": [
        {
            "title": "プログラミング言語って何？",
            "description": "コンピューターに「こう動いてね」とお願いするための特別な言葉、それがプログラミング言語です。人間とコンピューターの間に立って、お互いの言葉を通訳してくれる「橋」のようなものだと考えてみましょう。このレッスンでは、その橋の役割や、問題を解決するための手順書である「アルゴリズム」について学びます。また、昔のコンピューターで使われていた言葉から、現代のJavaに至るまでの壮大な歴史をたどります。\nこのレッスンで学ぶこと：プログラミング言語が人間とコンピューターのコミュニケーションに不可欠なツールであることを理解します。",
            "img_desc": {
                "positive": "A cute cartoon brain character and a robot character shaking hands over a glowing digital bridge, symbolizing communication, friendly style.",
                "negative": "ugly, blurry, low-resolution, text, watermark"
            },
            "book_page_start": 1,
            "book_page_end": 17,
            "key_vocabulary": [
                {
                    "jp": "プログラミング言語"
                },
                {
                    "jp": "アルゴリズム"
                },
                {
                    "jp": "コンピュータ"
                }
            ],
            "activity_suggestion": "友達に「ジュースを飲む」という動きを、一つ一つの命令に分解して伝えてみよう！これがアルゴリズムの考え方だよ。",
            "quizzes": [
                {
                    "title": "プログラミング言語とは、何と何を繋ぐ「橋」のようなものと説明されていますか？",
                    "book_page_start": 1,
                    "book_page_end": 17,
                    "options": [
                        "A: 先生と生徒",
                        "B: 人とコンピュータ",
                        "C: 親と子",
                        "D: 日本とアメリカ"
                    ],
                    "answer": "B",
                    "explanation": "プログラミング言語は、人間が考えた命令をコンピュータが理解できる形に翻訳する、大切な橋渡しの役目をしています。"
                }
            ]
        },
        {
            "title": "Javaの誕生と歴史",
            "description": "今から約30年前、インターネットが世界中に広がり始めた頃、新しい時代のヒーローとして「Java」というプログラミング言語が誕生しました。開発したのはサン・マイクロシステムズ社のジェームス・ゴスリンさんたち。彼らの目標は、パソコンだけでなく、家電や携帯電話など、どんな種類の機械でも動くプログラムを作ることでした。この「どこでも動く」という素晴らしい特徴が、Javaを世界的なスターにしたのです。\nこのレッスンで学ぶこと：Javaが特定の機械に縛られない、汎用性の高い言語として開発された背景を学びます。",
            "img_desc": {
                "positive": "A cartoon illustration of a superhero character with a Java coffee cup logo on its chest, flying over various devices like a laptop, smartphone, and TV.",
                "negative": "ugly, blurry, low-resolution, text, watermark"
            },
            "book_page_start": 1,
            "book_page_end": 17,
            "key_vocabulary": [
                {
                    "jp": "プラットフォーム"
                },
                {
                    "jp": "インターネット"
                },
                {
                    "jp": "開発"
                }
            ],
            "activity_suggestion": "身の回りにある色々な機械（スマホ、テレビ、ゲーム機）が、もしかしたらJavaで動いているかも？調べてみよう！",
            "quizzes": [
                {
                    "title": "Java言語を開発した中心人物は誰ですか？",
                    "book_page_start": 1,
                    "book_page_end": 17,
                    "options": [
                        "A: ビル・ゲイツ",
                        "B: スティーブ・ジョブズ",
                        "C: ジェームス・ゴスリン",
                        "D: イーロン・マスク"
                    ],
                    "answer": "C",
                    "explanation": "ジェームス・ゴスリンは「Javaの父」として知られており、開発チームの中心的な役割を果たしました。"
                }
            ]
        },
        {
            "title": "Javaのすごいところ！",
            "description": "Javaの一番の強みは「プラットフォームに依存しない」ことです。これはどういうことかというと、「Java仮想マシン（JVM）」という通訳者がいれば、WindowsのパソコンでもMacでも、どんなコンピューターでも同じプログラムが動くのです。プログラムを書く人は、一度コードを書くだけ。このJVMが、それぞれの機械に合わせた言葉に翻訳してくれるおかげで、開発がとても楽になります。これがJavaの魔法の秘密です。\nこのレッスンで学ぶこと：JVMの役割と、それがJavaの「一度書けば、どこでも動く」という特徴を実現している仕組みを理解します。",
            "img_desc": {
                "positive": "A cartoon magical machine labeled 'JVM' translating a single scroll of code into different shapes for a Windows-themed castle, a Mac-themed treehouse, and a Linux-themed spaceship.",
                "negative": "ugly, blurry, low-resolution, text, watermark"
            },
            "book_page_start": 1,
            "book_page_end": 17,
            "key_vocabulary": [
                {
                    "jp": "Java仮想マシン"
                },
                {
                    "jp": "バイトコード"
                },
                {
                    "jp": "オブジェクト指向"
                }
            ],
            "activity_suggestion": "日本語を英語や中国語に翻訳するアプリを使ってみよう。JVMも、Javaのコードを機械の言葉に翻訳する、すごい通訳者なんだ！",
            "quizzes": [
                {
                    "title": "Javaプログラムをどんなコンピュータでも動かせるようにする「通訳者」のようなソフトウェアは何ですか？",
                    "book_page_start": 1,
                    "book_page_end": 17,
                    "options": [
                        "A: JVK",
                        "B: JDK",
                        "C: JVM",
                        "D: JRE"
                    ],
                    "answer": "C",
                    "explanation": "JVMはJava Virtual Machine（Java仮想マシン）の略で、Javaバイトコードを各OSで実行できるようにする重要な役割を持っています。"
                }
            ]
        },
        {
            "title": "プログラミングの準備をしよう",
            "description": "Javaプログラミングを始めるには、まず「JDK（Java Development Kit）」という道具箱をパソコンにインストールする必要があります。Kitは「道具一式」という意味で、この中にはプログラムを作るためのコンパイラ（翻訳機）や、プログラムを動かすための実行環境（JRE）など、必要なものがすべて入っています。インストールが終わったら、コンピューターに「Javaの道具箱はここにあるよ」と教えてあげる「環境変数」の設定も行います。これで準備万端です！\nこのレッスンで学ぶこと：Java開発に必要なJDKの役割を理解し、インストールと基本的な環境設定の手順を学びます。",
            "img_desc": {
                "positive": "A cartoon toolbox labeled 'JDK' open, with tools like a hammer labeled 'Compiler' and a screwdriver labeled 'JRE' neatly arranged inside.",
                "negative": "ugly, blurry, low-resolution, text, watermark"
            },
            "book_page_start": 1,
            "book_page_end": 17,
            "key_vocabulary": [
                {
                    "jp": "JDK"
                },
                {
                    "jp": "インストール"
                },
                {
                    "jp": "環境変数"
                }
            ],
            "activity_suggestion": "自分の部屋で、よく使うものを決まった場所に置くように整理整頓してみよう。環境変数の設定も、コンピューターに道具の場所を教える整理整頓と似ているよ。",
            "quizzes": [
                {
                    "title": "Javaプログラムを開発するために必要な道具一式は何と呼ばれますか？",
                    "book_page_start": 1,
                    "book_page_end": 17,
                    "options": [
                        "A: Java Lunch Box (JLB)",
                        "B: Java Development Kit (JDK)",
                        "C: Java Gaming System (JGS)",
                        "D: Java Music Player (JMP)"
                    ],
                    "answer": "B",
                    "explanation": "JDKはJava Development Kitの略で、Javaでプログラムを開発するために必要なツールがすべて含まれています。"
                }
            ]
        },
        {
            "title": "はじめてのJavaプログラム",
            "description": "さあ、いよいよ自分の手でプログラムを書いてみましょう！最初は、画面に「こんにちは！」というメッセージを表示させる簡単なプログラムから始めます。テキストエディタ（メモ帳など）に決まったおまじないの言葉（コード）を書き、`MyFirstProgram.java`のように名前を付けて保存します。次に、コマンドプロンプトという黒い画面で「コンパイル」という翻訳作業を行うと、コンピューターが読める`.class`ファイルが完成。最後にそれを実行すれば、あなたのメッセージが画面に表示されます！\nこのレッスンで学ぶこと：Javaプログラム作成の基本的な流れ（コーディング、コンパイル、実行）を体験します。",
            "img_desc": {
                "positive": "A sequence of three cartoon scenes: 1. A child writing code on a notepad. 2. A magic wand labeled 'javac' turning the notepad into a scroll. 3. The child proudly pointing to a screen showing 'Hello!'",
                "negative": "ugly, blurry, low-resolution, text, watermark"
            },
            "book_page_start": 1,
            "book_page_end": 17,
            "key_vocabulary": [
                {
                    "jp": "クラス"
                },
                {
                    "jp": "メソッド"
                },
                {
                    "jp": "コンパイル"
                }
            ],
            "activity_suggestion": "「Welcome to ApexSoft professional training!」のメッセージを、自分の名前や「プログラミングは楽しい！」などの好きな言葉に変えて実行してみよう！",
            "quizzes": [
                {
                    "title": "`MyFirstProgram.java` というファイルをコンパイルすると、何というファイルが作成されますか？",
                    "book_page_start": 1,
                    "book_page_end": 17,
                    "options": [
                        "A: MyFirstProgram.txt",
                        "B: MyFirstProgram.exe",
                        "C: MyFirstProgram.class",
                        "D: MyFirstProgram.jpg"
                    ],
                    "answer": "C",
                    "explanation": "Javaコンパイラは `.java` というソースファイルをコンパイルして、JVMが実行できる `.class` というバイトコードファイルを作成します。"
                }
            ]
        },
        {
            "title": "もっと挑戦してみよう！",
            "description": "最初のプログラムが動いたら、次はもっと色々なことに挑戦してみましょう！このレッスンの最後には、練習問題がたくさん用意されています。例えば、アスタリスク（*）の文字を使って画面に三角形や四角形を描いてみたり、自分のプロフィール（名前、好きなものなど）を表示するプログラムを作ったり。また、改行やタブといった特別な意味を持つ文字の使い方も学びます。これらの課題に挑戦することで、プログラミングの基礎がしっかりと身につきます。\nこのレッスンで学ぶこと：簡単な図形描画や文字列表現を通して、基本的なプログラミング技術を応用する練習をします。",
            "img_desc": {
                "positive": "A cartoon child looking excitedly at a computer screen that displays various shapes (a star, a triangle, a square) made out of asterisk characters.",
                "negative": "ugly, blurry, low-resolution, text, watermark"
            },
            "book_page_start": 1,
            "book_page_end": 17,
            "key_vocabulary": [
                {
                    "jp": "課題"
                },
                {
                    "jp": "挑戦"
                },
                {
                    "jp": "図形"
                }
            ],
            "activity_suggestion": "アスタリスク（*）だけでなく、プラス（+）やマイナス（-）など、他の記号を使っても模様を描けるか試してみよう！",
            "quizzes": [
                {
                    "title": "画面上で文字を表示した後に、自動で次の行に移動する命令はどちらですか？",
                    "book_page_start": 1,
                    "book_page_end": 17,
                    "options": [
                        "A: print()",
                        "B: draw()",
                        "C: println()",
                        "D: write()"
                    ],
                    "answer": "C",
                    "explanation": "`println()`は 'print line' の略で、文字を表示した後に改行します。`print()`は改行しません。"
                }
            ]
        }
    ],
    "quizzes": [
        {
            "title": "人間がコンピュータに指示を与えるために使う、特別なルールの集まりを何と呼びますか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: アルゴリズム",
                "B: プログラム",
                "C: プログラミング言語",
                "D: インターネット"
            ],
            "answer": "C",
            "explanation": "プログラミング言語は、人間とコンピュータがコミュニケーションをとるための「言葉」や「文法」のルールです。",
            "paragraph_reference": "Програмчлалын хэл бол програмыг гүйцэтгэх командуудыг хүн болон машинд"
        },
        {
            "title": "1970年代に登場し、後のオブジェクト指向プログラミングに大きな影響を与えた言語は何ですか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: FORTRAN",
                "B: BASIC",
                "C: C, Pascal",
                "D: Java"
            ],
            "answer": "C",
            "explanation": "1970年代にはC言語やPascalが登場し、プログラムの構造化に大きな進歩をもたらしました。これが後のC++などの基礎となります。",
            "paragraph_reference": "1970-аад он. Энэ үед C, Pascal зэрэг маш хүчирхэг олон програмчлалын хэлнүүд гарсан"
        },
        {
            "title": "Javaが開発された当初の主な目的は何でしたか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: Webサイトを速くするため",
                "B: 様々な種類の電子機器で動くプログラムを作るため",
                "C: ゲームを開発するため",
                "D: AIを開発するため"
            ],
            "answer": "B",
            "explanation": "Javaは当初、パソコンだけでなく、家電や携帯情報端末（PDA）など、様々なデバイスで動作するプラットフォーム非依存の言語を目指して開発されました。",
            "paragraph_reference": "Java-г анх олон үйлдлийн систем, платформ үл хамаарах хэл болгох"
        },
        {
            "title": "Javaのソースコードをコンパイルした後にできる、JVMが読み取るファイル形式は何ですか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: ソースコード",
                "B: マシンコード",
                "C: バイトコード",
                "D: オブジェクトコード"
            ],
            "answer": "C",
            "explanation": "JavaコンパイラはJavaのソースコードを「バイトコード」に変換します。このバイトコードをJVMが解釈して、各コンピュータで実行します。",
            "paragraph_reference": "Java объект код/байт код"
        },
        {
            "title": "Javaのプログラムを実行するために使うコマンドは何ですか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: javac",
                "B: java",
                "C: run",
                "D: execute"
            ],
            "answer": "B",
            "explanation": "コンパイルには `javac` コマンドを、コンパイル済みの `.class` ファイルを実行するには `java` コマンドを使います。",
            "paragraph_reference": ">java MyFirstProgram"
        },
        {
            "title": "Javaのプログラムが必ず最初に実行する、特別な名前のメソッドは何ですか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: start()",
                "B: run()",
                "C: main()",
                "D: begin()"
            ],
            "answer": "C",
            "explanation": "`main` メソッドは、Javaプログラムの入り口（エントリーポイント）として定められており、プログラムを実行するとJVMが最初にこのメソッドを呼び出します。",
            "paragraph_reference": "Програмын үндсэн функц нь програмыг хамгийн анх ажиллахад дуудагдах функц"
        },
        {
            "title": "クラスの名前とソースファイルの名前は、どういう関係にあるのが望ましいですか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: 全く違う名前にする",
                "B: 同じ名前にする",
                "C: ファイル名を長くする",
                "D: 関係ない"
            ],
            "answer": "B",
            "explanation": "`public` なクラスを定義する場合、クラス名と`.java`のファイル名は一致させる必要があります。これはJavaの重要なルールです。",
            "paragraph_reference": "Классын нэр нь .java файлын нэртэйгээ ижил байдаг гэдгийг"
        },
        {
            "title": "コンピュータの画面に文字列を表示し、改行するJavaのコードはどれですか？",
            "book_page_start": 1,
            "book_page_end": 17,
            "options": [
                "A: Console.write(\"...\")",
                "B: System.out.println(\"...\")",
                "C: Screen.display(\"...\")",
                "D: print(\"...\")"
            ],
            "answer": "B",
            "explanation": "`System.out.println()` は、標準出力（通常はコンソール画面）に引数の文字列を表示した後に、改行を行うためのメソッドです。",
            "paragraph_reference": "System.out.println(\"Welcome to ApexSoft professional"
        }
    ],
    "mindmap": [
        {
            "title": "Javaプログラミング入門",
            "children": [
                {
                    "title": "プログラミングの基本",
                    "children": [
                        {
                            "title": "プログラミング言語とは？"
                        },
                        {
                            "title": "プログラミングの歴史（アセンブリからJavaまで）"
                        },
                        {
                            "title": "アルゴリズムの概念"
                        }
                    ]
                },
                {
                    "title": "Java言語の紹介",
                    "children": [
                        {
                            "title": "Javaの誕生と歴史"
                        },
                        {
                            "title": "Javaを学ぶ理由"
                        },
                        {
                            "title": "Javaの特徴と利点",
                            "children": [
                                {
                                    "title": "プラットフォーム非依存"
                                },
                                {
                                    "title": "オブジェクト指向"
                                },
                                {
                                    "title": "Java仮想マシン（JVM）の役割"
                                }
                            ]
                        }
                    ]
                },
                {
                    "title": "開発環境の準備",
                    "children": [
                        {
                            "title": "JDK（Java Development Kit）とは？"
                        },
                        {
                            "title": "JDKのインストール手順"
                        },
                        {
                            "title": "環境変数の設定（Path）"
                        },
                        {
                            "title": "インストールの確認（javacコマンド）"
                        }
                    ]
                },
                {
                    "title": "最初のプログラム",
                    "children": [
                        {
                            "title": "コードの作成（Notepad）"
                        },
                        {
                            "title": "コンパイル（javacコマンド）"
                        },
                        {
                            "title": "実行（javaコマンド）"
                        },
                        {
                            "title": "コードの構造解説",
                            "children": [
                                {
                                    "title": "クラス宣言"
                                },
                                {
                                    "title": "mainメソッド"
                                },
                                {
                                    "title": "System.out.println"
                                }
                            ]
                        }
                    ]
                },
                {
                    "title": "次のステップ（練習問題）",
                    "children": [
                        {
                            "title": "printとprintlnの違い"
                        },
                        {
                            "title": "図形の描画"
                        },
                        {
                            "title": "特殊文字の利用"
                        }
                    ]
                }
            ]
        }
    ],
    "audio_lessons": [
        {
            "title": "プログラミング言語ってなあに？",
            "book_page_start": 1,
            "book_page_end": 17,
            "length_minutes": 3,
            "key_points": [
                "プログラミング言語は、人間とコンピュータが会話するための言葉です。",
                "コンピュータは、人間が書いたプログラムという「指示書」の通りに動きます。",
                "昔からたくさんの種類のプログラミング言語が作られてきました。"
            ],
            "script": "こんにちは！プログラミングの世界へようこそ。みんなは、外国の人と話すとき、英語や中国語といった「言葉」を使うよね。それと同じで、コンピュータに何かをしてもらいたいときにも、コンピュータがわかる「言葉」が必要なんだ。それが「プログラミング言語」だよ。プログラミング言語を使って、「この計算をしてね」とか「この絵を表示してね」という指示書を書くこと、これがプログラミングなんだ。このレッスンでは、その中でも特に人気のJavaという言語について学んでいくよ。楽しみにしていてね！"
        },
        {
            "title": "Javaはなぜ特別なの？JVMのひみつ",
            "book_page_start": 1,
            "book_page_end": 17,
            "length_minutes": 4,
            "key_points": [
                "Javaは「一度書けば、どこでも動く」が合言葉です。",
                "「Java仮想マシン（JVM）」という通訳のおかげで、どんなコンピュータでも動きます。",
                "プログラムを一度書くだけで良いので、開発がとても効率的になります。"
            ],
            "script": "Javaが世界中で使われている大きな理由、それは「一度書けば、どこでも動く」という魔法のような力があるからなんだ。例えば、Windowsのパソコン用に作ったプログラムが、Macのパソコンでも、スマホでもそのまま動いちゃうんだ。すごいよね！この魔法の秘密は「Java仮想マシン」、略してJVMというソフトウェアにある。JVMは、Javaのプログラムをそれぞれの機械がわかる言葉に翻訳してくれる、とっても賢い通訳さんのようなものなんだ。だから、プログラマーは機械の種類を気にせず、プログラム作りに集中できるんだよ。"
        },
        {
            "title": "はじめてのプログラムを作ろう！",
            "book_page_start": 1,
            "book_page_end": 17,
            "length_minutes": 5,
            "key_points": [
                "まず、メモ帳などのテキストエディタに決まった形式でコードを書きます。",
                "次に、`javac` というコマンドでプログラムを「コンパイル（翻訳）」します。",
                "最後に、`java` というコマンドでプログラムを「実行」して結果を見ます。"
            ],
            "script": "さあ、いよいよ君もプログラマーだ！最初のプログラムを作ってみよう。まずはメモ帳を開いて、教科書に書いてある通りに`public class MyFirstProgram`から始まるコードを打ち込んでみて。一文字も間違えないように気をつけてね。書き終わったら、「MyFirstProgram.java」という名前で保存しよう。次に、コマンドプロンプトという黒い画面を開いて、「javac MyFirstProgram.java」と打ってエンター！これで翻訳完了。最後に、「java MyFirstProgram」と打ってみて。画面にメッセージが表示されたら大成功だよ！おめでとう！"
        },
        {
            "title": "クラスとmainメソッド",
            "book_page_start": 1,
            "book_page_end": 17,
            "length_minutes": 3,
            "key_points": [
                "Javaのプログラムは、「クラス」という設計図の中に書かれます。",
                "`main` メソッドは、プログラムが動き出す「スタート地点」です。",
                "プログラムの実行は、必ず `main` メソッドから始まります。"
            ],
            "script": "最初のプログラム、ちょっと呪文みたいで難しかったかな？少しだけ解説するね。Javaのプログラムは、まず「クラス」という大きな箱の中に作るんだ。今回の箱の名前は `MyFirstProgram` だったね。そして、その箱の中に `main` という名前の特別な場所がある。ここがプログラムのスタート地点なんだ。コンピュータはプログラムを動かすとき、まずこの `main` という場所を探して、そこから指示を読み始めるんだよ。だから、Javaプログラムには必ずこの `main` が必要になるって覚えておこう！"
        }
    ]
};
// async function generateAudio(text) {
//     const response = await ai.models.generateContent({
//         model: "gemini-2.5-flash-preview-tts",
//         contents: [{ parts: [{ text: text }] }],
//         config: {
//             responseModalities: ['AUDIO'],
//             speechConfig: {
//                 voiceConfig: {
//                     prebuiltVoiceConfig: { voiceName: 'Kore' },
//                 },
//             },
//         },
//     });
//     const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
//     const audioBuffer = Buffer.from(data, 'base64');
//     const fileName = `${v4()}.wav`;
//     const publicPath = path.join(process.cwd(), 'public', fileName);
//     await saveWaveFile(publicPath, audioBuffer);
//     return fileName;
// }
// async function generate_image(originalPrompt, attempt = 0) {
//     // 2-оос дээш удаа оролдвол зогсооно (Эхний оролдлого: 0, Давтан: 1)
//     if (attempt > 1) {
//         console.error("Failed to generate image after multiple attempts.");
//         return null;
//     }
//     let promptToSend = originalPrompt;
//     // Эхний оролдлого биш бол prompt-д нэмэлт мэдээлэл оруулна
//     if (attempt > 0) {
//         promptToSend += ", clear image, high quality, no text, no watermark";
//         console.log("Retrying with modified prompt...");
//     }
//     console.log(`Generating image with prompt (attempt ${attempt + 1}): "${promptToSend}"`);
//     try {
//         const response = await ai.models.generateContent({
//             model: "gemini-2.5-flash-image",
//             // 1. API-д шаардлагатай зөв форматаар илгээх
//             contents: [{
//                 parts: [{
//                     text: promptToSend
//                 }]
//             }],
//             config: {
//                 candidateCount: 1,
//                 useCache: false
//             }
//         });
//         // 3. Хариу хоосон эсэхийг шалгах
//         if (!response.candidates || response.candidates.length === 0) {
//             console.log("API returned no candidates. Retrying...");
//             // Дахин оролдохдоо prompt-ийг өөрчилнө
//             return generate_image(originalPrompt, attempt + 1);
//         }
//         const part = response.candidates[0].content.parts[0];
//         if (part.inlineData) {
//             const imageData = part.inlineData.data;
//             const buffer = Buffer.from(imageData, "base64");
//             const fileName = `${v4()}.png`;
//             const publicPath = path.join(process.cwd(), 'public', fileName);
//             await fs.writeFile(publicPath, buffer);
//             console.log(`Image saved successfully as ${fileName}`);
//             return fileName;
//         } else if (part.text) {
//             console.log(`Model returned text instead of image. Reason: "${part.text}". Retrying...`);
//             // 2. Логикийн алдааг засах: Дахин оролдох
//             return generate_image(originalPrompt, attempt + 1);
//         } else {
//             console.log("Unexpected response part format. Retrying...");
//             return generate_image(originalPrompt, attempt + 1);
//         }
//     } catch (error) {
//         // 3. API дуудалтын алдааг барих
//         console.error("An error occurred during image generation:", error);
//         return null;
//     }
// }
// async function saveWaveFile(
//     filename,
//     pcmData,
//     channels = 1,
//     rate = 24000,
//     sampleWidth = 2,
// ) {
//     return new Promise((resolve, reject) => {
//         const writer = new wav.FileWriter(filename, {
//             channels,
//             sampleRate: rate,
//             bitDepth: sampleWidth * 8,
//         });
//         writer.on('finish', resolve);
//         writer.on('error', reject);
//         writer.write(pcmData);
//         writer.end();
//     });
// }
// async function generate_video(prompt) {
//     let operation = await ai.models.generateVideos({
//         model: "veo-3.0-generate-001",
//         prompt: prompt,
//     });
//     // Poll the operation status until the video is ready.
//     while (!operation.done) {
//         console.log("Waiting for video generation to complete...")
//         await new Promise((resolve) => setTimeout(resolve, 10000));
//         operation = await ai.operations.getVideosOperation({
//             operation: operation,
//         });
//     }
//     console.log(JSON.stringify(operation.response))
//     // Download the generated video.
//     ai.files.download({
//         file: operation.response.generatedVideos[0].video,
//         downloadPath: "./dialogue_example.mp4",
//     });
//     return "./dialogue_example.mp4";
// }
