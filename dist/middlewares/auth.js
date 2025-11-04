"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.supabaseAuth = supabaseAuth;
const supabase_1 = __importDefault(require("../lib/supabase"));
const supabase_js_1 = require("@supabase/supabase-js");
async function supabaseAuth(req, res, next) {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ error: "No token provided" });
    }
    const token = authHeader.split(" ")[1];
    // Validate token and get user
    const { data: user, error } = await supabase_1.default.auth.getUser(token);
    if (error || !user) {
        return res.status(401).json({ error: "Invalid token" });
    }
    req.user = user; // inject user info into request
    // Token-ийг тухайн хэрэглэгчийн session-д зориулж client-д оруулна
    req.supabase = (0, supabase_js_1.createClient)(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY, {
        global: {
            headers: { Authorization: `Bearer ${token}` },
        },
    });
    next();
}
