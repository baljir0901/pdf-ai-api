"use strict";
// TEMPORARY FIX: Commented out due to ESM import issue in Vercel serverless
// This function is only used by /user/upload-file endpoint which is not critical
// import { ai_lesson_generate } from "../lib/ai_lessons"
const Router = require("express").Router;
const express = require("express");
const path = require("path");
const fs = require("fs");
const http = require("http");
const https = require("https");
const { pipeline } = require("stream");
const { promisify } = require("util");
const { ingestPdfToVectorDB, askQuestion, ingestPdfWithVision } = require("../lib/pdf");
const { supabaseAuth } = require("../middlewares/auth");
const { ai_lesson_generate } = require("../lib/ai_lessons");
const supabase = require("../lib/supabase").default;
const streamPipeline = promisify(pipeline);
const router = Router();
const uploadDir = "/tmp/uploads"; // зөвхөн энэ хавтас руу бичиж болно
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}
// max file size (bytes) — өөрчилж болно (энд 50 MB)
const MAX_BYTES = 50 * 1024 * 1024;
// зөвшөөрөх content-type
const ALLOWED_TYPES = ["application/pdf"];
async function downloadPdfWithRedirects(fileUrl, destPath, maxRedirects = 5) {
    return new Promise((resolve, reject) => {
        let redirects = 0;
        function doRequest(urlStr) {
            const urlObj = new URL(urlStr);
            const client = urlObj.protocol === "https:" ? https : http;
            const req = client.get(urlObj, { headers: { "User-Agent": "Node.js PDF Downloader" } }, (res) => {
                const status = res.statusCode;
                // Follow redirects (301/302/307/308)
                if (status >= 300 && status < 400 && res.headers.location) {
                    if (redirects++ >= maxRedirects) {
                        return reject(new Error("Too many redirects"));
                    }
                    // follow location (may be relative)
                    const location = new URL(res.headers.location, urlObj).toString();
                    // consume and discard current stream before following
                    res.resume();
                    return doRequest(location);
                }
                if (status !== 200) {
                    res.resume();
                    return reject(new Error(`Request Failed. Status Code: ${status}`));
                }
                // check headers for pdf
                const contentType = res.headers["content-type"] || "";
                const contentLength = parseInt(res.headers["content-length"] || "0", 10);
                // If content-length present, check size limit early
                if (contentLength && contentLength > MAX_BYTES) {
                    res.resume();
                    return reject(new Error(`File too large (Content-Length ${contentLength} bytes)`));
                }
                // allow if header says pdf, otherwise we'll still stream but will validate first chunk
                if (!ALLOWED_TYPES.some((t) => contentType.toLowerCase().includes(t))) {
                    // not immediately reject — some servers send wrong content-type.
                    // but we will still allow and check magic bytes? Simpler: reject to be strict.
                    res.resume();
                    return reject(new Error(`Invalid content-type: ${contentType}. Only PDFs allowed.`));
                }
                // stream to file and enforce byte limit while downloading
                const fileStream = fs.createWriteStream(destPath, { flags: "w" });
                let downloaded = 0;
                let aborted = false;
                res.on("data", (chunk) => {
                    downloaded += chunk.length;
                    if (downloaded > MAX_BYTES) {
                        aborted = true;
                        req.destroy();
                        fileStream.destroy();
                        // delete partial file
                        try {
                            fs.unlinkSync(destPath);
                        }
                        catch (e) { }
                        return reject(new Error("Download aborted: file size exceeds limit"));
                    }
                });
                // pipe response to file
                streamPipeline(res, fileStream)
                    .then(() => {
                    if (!aborted)
                        resolve({ path: destPath, bytes: downloaded, contentType });
                })
                    .catch((err) => {
                    // cleanup partial file
                    try {
                        fs.unlinkSync(destPath);
                    }
                    catch (e) { }
                    reject(err);
                });
            });
            req.on("error", (err) => {
                reject(err);
            });
            // optional timeout (e.g., 30s)
            req.setTimeout(30_000, () => {
                req.abort();
                reject(new Error("Request timeout"));
            });
        }
        doRequest(fileUrl);
    });
}
// Test endpoint - GET /api/v1/chat-test
router.get("/chat-test", (req, res) => {
    console.log("✅ [PDF-AI-API] GET /api/v1/chat-test endpoint working");
    res.json({
        ok: true,
        message: "Chat endpoint is accessible",
        note: "Use POST /api/v1/chat with {question, id} in body",
        timestamp: new Date().toISOString()
    });
});
router.post("/chat", async (req, res) => {
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    console.log("💬 [PDF-AI-API] POST /api/v1/chat");
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    console.log("📦 Request Body:", JSON.stringify(req.body, null, 2));
    console.log("🔑 Headers:", JSON.stringify(req.headers, null, 2));
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    try {
        const { question, id, conversationId } = req.body;
        if (!question) {
            console.error("❌ Missing required parameters");
            return res.status(400).json({
                ok: false,
                error: "question and id are required"
            });
        }
        console.log("✓ Calling askQuestion with:", { question: question.substring(0, 50) + "...", id, conversationId });
        const result = await askQuestion(question, id, id, conversationId, req.body.pdfUrl, req.body.currentPage);
        console.log("✅ Chat response generated successfully");
        console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        res.json(result);
    }
    catch (error) {
        console.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        console.error("❌ Chat Error:", error);
        console.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        res.status(500).json({
            ok: false,
            error: error.message || "Chat failed",
            details: error.toString()
        });
    }
});
router.post("/upload-pdf", async (req, res) => {
    console.log(req.body);
    try {
        const url = req.body && req.body.pdf_url;
        const useVision = req.body && Boolean(req.body.use_vision) === true; // Optional: enable Vision mode
        if (!url)
            return res.status(400).json({ ok: false, error: "pdf_url is required in JSON body" });
        // validate URL
        let urlObj;
        try {
            urlObj = new URL(url);
            if (!["http:", "https:"].includes(urlObj.protocol)) {
                return res.status(400).json({ ok: false, error: "Only http/https URLs are allowed" });
            }
        }
        catch (err) {
            return res.status(400).json({ ok: false, error: "Invalid URL" });
        }
        // build filename: try to use basename if .pdf, otherwise generate unique name
        let basename = path.basename(urlObj.pathname) || "file.pdf";
        // if no .pdf extension then append .pdf for storage filename (we still validate content-type)
        if (!basename.toLowerCase().endsWith(".pdf"))
            basename = basename + ".pdf";
        // sanitize basename: remove query params etc and unsafe chars
        basename = basename.replace(/[^a-zA-Z0-9._-]/g, "_");
        const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1e9)}-${basename}`;
        const destPath = path.join(uploadDir, uniqueName);
        // download (follows limited redirects, validates content-type and size)
        const result = await downloadPdfWithRedirects(url, destPath, 5);
        // Choose extraction method
        let ingestionResult;
        if (useVision) {
            console.log("🔮 Using Gemini Vision for PDF extraction (with images/tables)");
            ingestionResult = await ingestPdfWithVision(destPath, req.body.id);
        }
        else {
            console.log("📄 Using standard text-only extraction");
            ingestionResult = await ingestPdfToVectorDB(destPath, req.body.id);
        }
        // success
        return res.json({
            ok: true,
            message: useVision ? "PDF downloaded and processed with Vision" : "PDF downloaded",
            file: {
                filename: uniqueName,
                path: destPath,
                bytes: result.bytes,
                contentType: result.contentType,
            },
            ingestion: ingestionResult, // Include vectorization result
        });
    }
    catch (err) {
        console.error("Download error:", err);
        return res.status(400).json({ ok: false, error: err.message || "Download failed" });
    }
});
router.post("/user/upload-file", supabaseAuth, async (req, res) => {
    try {
        const url = req.body && req.body.file_url;
        console.log(req.user, url);
        if (!url)
            return res.status(400).json({ ok: false, error: "url is required in JSON body" });
        // validate URL
        let urlObj;
        try {
            urlObj = new URL(url);
            if (!["http:", "https:"].includes(urlObj.protocol)) {
                return res.status(400).json({ ok: false, error: "Only http/https URLs are allowed" });
            }
        }
        catch (err) {
            return res.status(400).json({ ok: false, error: "Invalid URL" });
        }
        // build filename: try to use basename if .pdf, otherwise generate unique name
        let basename = path.basename(urlObj.pathname) || "file.pdf";
        // if no .pdf extension then append .pdf for storage filename (we still validate content-type)
        if (!basename.toLowerCase().endsWith(".pdf"))
            basename = basename + ".pdf";
        // sanitize basename: remove query params etc and unsafe chars
        basename = basename.replace(/[^a-zA-Z0-9._-]/g, "_");
        const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1e9)}-${basename}`;
        const destPath = path.join(uploadDir, uniqueName);
        const result = await downloadPdfWithRedirects(url, destPath, 5);
        res.json(await ai_lesson_generate(destPath, req.supabase, url));
    }
    catch (err) {
        console.log(err);
        res.status(500).json(err);
    }
});
module.exports = router;
