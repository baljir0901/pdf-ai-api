"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const genai_1 = require("@google/genai");
const gemini_ai = new genai_1.GoogleGenAI({
    apiKey: process.env.GOOGLE_API_KEY
});
exports.default = gemini_ai;
