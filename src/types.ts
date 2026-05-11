export interface TickerData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  exchange: 'HOSE' | 'HNX' | 'UPCOM';
}

export interface Position {
  id: string;
  symbol: string;
  avgPrice: number;
  quantity: number;
  currentPrice: number;
  entryDate: string;
  pnl: number;
  pnlPercent: number;
}

export interface Trade {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  timestamp: string;
  pnl?: number;
  aiConfidence?: number;
  feedback?: 'SUCCESS' | 'FAILURE';
}

export interface LearningLesson {
  symbol: string;
  outcome: 'SUCCESS' | 'FAILURE';
  reason: string;
  pnl: number;
}

export interface Signal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
  timestamp: string;
}

export interface PortfolioStats {
  balance: number;
  equity: number;
  totalPnL: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
}
