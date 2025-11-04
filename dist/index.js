"use strict";
/**
 * ═══════════════════════════════════════════════════════════════
 * PDF AI API Server
 * ═══════════════════════════════════════════════════════════════
 *
 * Express backend сервер - PDF файлуудыг Vector Database-д хадгалж,
 * AI ашиглан semantic search болон chat функцүүдийг үйлчилнэ.
 *
 * Гол функцүүд:
 * - PDF upload ба vectorization (Weaviate + Gemini Embeddings)
 * - AI Chat with PDF (RAG - Retrieval Augmented Generation)
 * - Semantic search through PDF content
 *
 * Tech Stack:
 * - Express.js (REST API)
 * - Weaviate (Vector Database)
 * - Google Gemini AI (Embeddings + Chat)
 * - Supabase (Database)
 * - LangChain (PDF processing)
 *
 * @author CodyGym3 Team
 * @version 1.0.0
 */
require("dotenv").config();
const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const cors = require("cors");
const morgan = require("morgan");
const pdf_upload = require("./routes/pdf-upload");
const pdf_extract = require("./routes/pdf-extract");
const app = express();
const PORT = process.env.PORT || 3001; // Changed from 5401 to 3001 for consistency
// === Middleware Setup ===
/**
 * CORS тохиргоо - Cross-Origin Resource Sharing
 * Frontend (localhost:3000) -аас API дуудах боломжтой болгох
 */
app.use(cors());
app.use(morgan("dev")); // HTTP request logging
/** JSON body parser - POST request-ийн body-г JSON болгон унших */
app.use(express.json());
/** URL-encoded body parser - Form submission-ууд унших */
app.use(express.urlencoded({ extended: true }));
// === Upload Directory Setup ===
/**
 * Upload folder үүсгэх (хэрэв байхгүй бол)
 * PDF файлууд энд түр хадгалагдана download хийсний дараа
 */
// === Multer Storage Configuration (Unused but kept for reference) ===
/**
 * Multer storage тохиргоо
 * Хэрэв multipart/form-data upload ашиглах бол энийг ашиглана
 * Одоогоор ашиглагдахгүй байна (URL-аас PDF татдаг)
 */
// === Routes ===
/**
 * API Routes mount
 * /api/v1/upload-pdf - PDF upload endpoint
 * /api/v1/chat - AI chat with PDF endpoint
 * /extract-pdf - PDF text extraction endpoint
 */
app.use("/api/v1", pdf_upload);
app.use("/", pdf_extract);
/**
 * Health check endpoint
 * Серверийн ажиллагааг шалгах (monitoring)
 */
app.get("/health", (req, res) => {
    res.json({
        status: "ok",
        message: "PDF AI API is running",
        timestamp: new Date().toISOString(),
        port: PORT
    });
});
// === Error Handler Middleware ===
/**
 * Global error handler
 * Бүх route-ын алдаануудыг энд барьж, хариулт өгнө
 */
app.use((err, req, res, next) => {
    console.error("❌ Server Error:", err);
    res.status(500).json({
        ok: false,
        error: err.message || "Internal server error"
    });
});
// === Server Start ===
app.listen(PORT, () => {
    console.log("\n╔═══════════════════════════════════════════════╗");
    console.log(`║  PDF AI API Server Started                    ║`);
    console.log(`║  Port: ${PORT}                                   ║`);
    console.log(`║  URL: http://localhost:${PORT}                   ║`);
    console.log("╚═══════════════════════════════════════════════╝\n");
});
