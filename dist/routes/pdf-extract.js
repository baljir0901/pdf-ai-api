"use strict";
/**
 * PDF Extract Route
 *
 * PDF файлаас текст татах endpoint
 * Google Gemini Vision API ашиглана
 */
const express = require('express');
const router = express.Router();
const { GoogleGenerativeAI } = require('@google/generative-ai');
// Gemini AI client (GOOGLE_API_KEY эсвэл GEMINI_API_KEY ашиглана)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '');
/**
 * POST /extract-pdf
 * PDF URL-аас текст татах
 *
 * Body: { pdfUrl: string }
 * Response: { text: string, success: boolean }
 */
router.post('/extract-pdf', async (req, res) => {
    try {
        const { pdfUrl } = req.body;
        if (!pdfUrl) {
            return res.status(400).json({ error: 'PDF URL required' });
        }
        if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
            return res.status(500).json({ error: 'Gemini API key not configured' });
        }
        console.log('📄 pdf-ai-api: Fetching PDF from:', pdfUrl);
        // PDF татах
        const pdfResponse = await fetch(pdfUrl);
        if (!pdfResponse.ok) {
            throw new Error('Failed to fetch PDF');
        }
        // Base64 болгох
        const arrayBuffer = await pdfResponse.arrayBuffer();
        const base64Data = Buffer.from(arrayBuffer).toString('base64');
        console.log('🤖 pdf-ai-api: Sending PDF to Gemini Vision API...');
        // Gemini модель
        const model = genAI.getGenerativeModel({
            model: 'gemini-2.5-flash-lite',
            generationConfig: {
                temperature: 0.2,
                topP: 0.95,
                topK: 40,
                maxOutputTokens: 4096,
                responseMimeType: "application/json",
            },
        });
        // PDF агуулга татах
        const result = await model.generateContent([
            {
                text: `このPDFの内容を正確に抽出してください。

【重要な指示】
1. すべてのセクションタイトルを抽出
2. すべての問題番号と問題内容を正確に
3. 数字、記号、すべてそのまま
4. 文章題も完全に
5. レイアウトや構造も保持

【出力形式】
セクション名を明記し、その下に問題を列挙してください。

例:
=== セクション1: たしざん (足し算) ===
1. 45 + 27 = ____
2. 63 - 28 = ____
3. 38 + 46 = ____

=== セクション2: かけざん (掛け算) ===
1. 6 × 4 = ____
2. 3 × 7 = ____

このような形式で、PDFのすべての内容を抽出してください。`,
            },
            {
                inlineData: {
                    mimeType: 'application/pdf',
                    data: base64Data,
                },
            },
        ]);
        const extractedText = result.response.text();
        console.log('✅ pdf-ai-api: PDF extraction complete!');
        console.log('📝 pdf-ai-api: Extracted text preview:', extractedText.substring(0, 200));
        res.json({
            text: extractedText,
            success: true,
        });
    }
    catch (error) {
        console.error('❌ pdf-ai-api: PDF extraction error:', error);
        res.status(500).json({
            error: 'Failed to extract PDF text',
            details: error.message,
        });
    }
});
module.exports = router;
