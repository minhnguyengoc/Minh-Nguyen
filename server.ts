import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  app.use((req, res, next) => {
    if (req.url.startsWith('/api')) {
      console.log(`[API] ${req.method} ${req.url}`);
    }
    next();
  });

  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  // === MOCK DATA & STATE ===
  let portfolio = {
    balance: 500000000, // 500M VND
    equity: 500000000,
    totalPnL: 0,
    winRate: 0,
    profitFactor: 0,
    totalTrades: 0
  };

  let positions: any[] = [];
  let tradeHistory: any[] = [];
  let learningLessons: any[] = [];
  
  // === API ROUTES ===
  app.get("/api/learning/lessons", (req, res) => {
    res.json(learningLessons);
  });

  app.get("/api/market/tickers", (req, res) => {
    // Market Tickers (HOSE Liquid Watchlist)
    const tickers = [
      { symbol: "VCB", price: 92400, change: 1200, changePercent: 1.3, volume: 1200000, exchange: "HOSE" },
      { symbol: "HPG", price: 28500, change: -450, changePercent: -1.5, volume: 15400000, exchange: "HOSE" },
      { symbol: "FPT", price: 115200, change: 3200, changePercent: 2.8, volume: 2100000, exchange: "HOSE" },
      { symbol: "SSI", price: 34100, change: 100, changePercent: 0.3, volume: 8900000, exchange: "HOSE" },
      { symbol: "VNM", price: 68500, change: -200, changePercent: -0.3, volume: 1800000, exchange: "HOSE" },
      { symbol: "VIC", price: 42100, change: -300, changePercent: -0.7, volume: 4500000, exchange: "HOSE" },
      { symbol: "TCB", price: 46200, change: 1500, changePercent: 3.3, volume: 6200000, exchange: "HOSE" },
    ];
    res.json(tickers);
  });

  app.get("/api/portfolio", (req, res) => {
    // Market Tickers (HOSE Liquid Watchlist) - Need same data for price calc
    const tickers = [
      { symbol: "VCB", price: 92400 },
      { symbol: "HPG", price: 28500 },
      { symbol: "FPT", price: 115200 },
      { symbol: "SSI", price: 34100 },
      { symbol: "VNM", price: 68500 },
      { symbol: "VIC", price: 42100 },
      { symbol: "TCB", price: 46200 },
    ];

    const currentPositionsValue = positions.reduce((acc, pos) => {
      const ticker = tickers.find(t => t.symbol === pos.symbol);
      return acc + (ticker ? ticker.price * pos.quantity : pos.avgPrice * pos.quantity);
    }, 0);
    
    portfolio.equity = portfolio.balance + currentPositionsValue;
    res.json({ ...portfolio, positions, history: tradeHistory });
  });

  app.post("/api/trading/feedback", (req, res) => {
    const { tradeId, feedback, reason, pnl } = req.body;
    const trade = tradeHistory.find(t => t.id === tradeId);
    if (trade) {
      trade.feedback = feedback;
      learningLessons.push({
        symbol: trade.symbol,
        outcome: feedback,
        reason: reason || "No description",
        pnl: pnl || 0,
        timestamp: new Date().toISOString()
      });
      res.json({ success: true, lessonsCount: learningLessons.length });
    } else {
      res.status(404).json({ error: "Trade not found" });
    }
  });

  app.post("/api/trading/execute", (req, res) => {
    const { type, symbol, quantity, price } = req.body;
    const totalCost = quantity * price;

    if (type === "BUY") {
      if (portfolio.balance < totalCost) return res.status(400).json({ error: "Insufficient balance" });
      
      portfolio.balance -= totalCost;
      const existingPos = positions.find(p => p.symbol === symbol);
      if (existingPos) {
        const totalQty = existingPos.quantity + quantity;
        existingPos.avgPrice = (existingPos.avgPrice * existingPos.quantity + totalCost) / totalQty;
        existingPos.quantity = totalQty;
      } else {
        positions.push({ id: Math.random().toString(36).substring(2, 11), symbol, avgPrice: price, quantity, currentPrice: price, pnl: 0, pnlPercent: 0 });
      }
    } else if (type === "SELL") {
      const pos = positions.find(p => p.symbol === symbol);
      if (!pos || pos.quantity < quantity) return res.status(400).json({ error: "Insufficient position" });

      const saleValue = quantity * price;
      portfolio.balance += saleValue;
      const pnl = (price - pos.avgPrice) * quantity;
      portfolio.totalPnL += pnl;
      
      pos.quantity -= quantity;
      if (pos.quantity === 0) {
        positions = positions.filter(p => p.symbol !== symbol);
      }
    }

    tradeHistory.push({ id: Math.random().toString(36).substring(2, 11), symbol, type, quantity, price, timestamp: new Date().toISOString() });
    res.json({ success: true, portfolio, positions });
  });

  // Handle missing API routes with JSON instead of HTML
  app.use("/api/*", (req, res) => {
    res.status(404).json({ error: `API route not found: ${req.method} ${req.originalUrl}` });
  });

  // === VITE MIDDLEWARE ===
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

startServer();
