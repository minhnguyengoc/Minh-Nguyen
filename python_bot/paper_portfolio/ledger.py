import os
import pandas as pd
import logging
import hashlib
from pathlib import Path
from python_bot.core.paths import PAPER_LOGS_DIR
from python_bot.core.exceptions import LedgerAccountingError

logger = logging.getLogger("VNStockBot.PaperLedgerCore")

class PaperLedger:
    """Core portfolio double-entry ledger to log assets and transactions consistently."""
    
    COLUMNS = [
        "order_id", "timestamp", "ticker", "action", "price", "qty",
        "cash_before", "cash_after", "position_before", "position_after",
        "avg_cost", "realized_pnl", "unrealized_pnl", "equity",
        "transaction_cost", "rejected_reason", "note", "status"
    ]
    
    def __init__(self, ticker: str, initial_cash: float = 100000000.0):
        self.ticker = ticker.upper()
        self.initial_cash = initial_cash
        self.file_path = PAPER_LOGS_DIR / f"paper_ledger_{self.ticker}_1m.csv"
        
        # State indicators
        self.cash = initial_cash
        self.position_qty = 0.0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0
        self.last_timestamp = None
        self.processed_order_ids = set()
        
        self._load_from_csv()

    def _load_from_csv(self):
        """Reconstruct state from historic records to ensure idempotency and resume capability."""
        self.cash = self.initial_cash
        self.position_qty = 0.0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0
        self.last_timestamp = None
        self.processed_order_ids = set()
        
        if self.file_path.exists():
            try:
                df = pd.read_csv(self.file_path)
                for _, row in df.iterrows():
                    o_id = row.get("order_id")
                    if pd.notna(o_id):
                        self.processed_order_ids.add(str(o_id))
                    
                    rejected_reason = row.get("rejected_reason")
                    is_rejected = pd.notna(rejected_reason) and str(rejected_reason).strip() != "" and str(rejected_reason).upper() != "NONE"
                    
                    status = row.get("status")
                    is_filled = pd.isna(status) or str(status).upper() == "FILLED"
                    
                    if is_filled and not is_rejected:
                        self.cash = float(row["cash_after"])
                        self.position_qty = float(row["position_after"])
                        self.avg_cost = float(row["avg_cost"]) if pd.notna(row["avg_cost"]) else 0.0
                        self.realized_pnl += float(row["realized_pnl"]) if pd.notna(row["realized_pnl"]) else 0.0
                        self.last_timestamp = row["timestamp"]
                        
                logger.info(f"Loaded ledger state: Cash={self.cash:.2f}, Shares={self.position_qty:.0f}, Cumulative Realized PnL={self.realized_pnl:.2f}, Processed Orders={len(self.processed_order_ids)}")
            except Exception as e:
                logger.error(f"Failed to restore ledger state: {e}")

    def record_entry(self, entry: dict):
        """Writes standard entries to the persistence csv representation atomically."""
        os.makedirs(PAPER_LOGS_DIR, exist_ok=True)
        row = {col: entry.get(col, None) for col in self.COLUMNS}
        new_df = pd.DataFrame([row])
        
        temp_file = self.file_path.with_suffix(".csv.tmp")
        if self.file_path.exists():
            try:
                df = pd.read_csv(self.file_path)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(temp_file, index=False)
                os.replace(temp_file, self.file_path)
            except Exception as e:
                logger.error(f"Failed to atomically write/append transaction ledger: {e}")
                if temp_file.exists():
                    os.remove(temp_file)
        else:
            try:
                new_df.to_csv(temp_file, index=False)
                os.replace(temp_file, self.file_path)
            except Exception as e:
                logger.error(f"Failed to atomically initialize transaction ledger: {e}")
                if temp_file.exists():
                    os.remove(temp_file)

    def process_order(self, timestamp: str, action: str, price: float, qty: float, transaction_cost: float = 0.0) -> dict:
        """
        Processes transaction order representation. Performs strict validation.
        Returns the recorded transaction entry dictionary.
        """
        if action is None:
            raise LedgerAccountingError("Action must be specified")
            
        action = action.upper()
        
        # 11. For SELL/CLOSE: If action == CLOSE, qty should default to full current position if qty is None or 0.
        if action == "CLOSE" and (qty is None or qty == 0.0):
            qty = self.position_qty
            
        # 10. Add hard validation
        if action not in ["BUY", "SELL", "CLOSE", "HOLD"]:
            raise LedgerAccountingError(f"Action must be in BUY, SELL, CLOSE, HOLD, got {action}")
            
        try:
            price = float(price)
        except (ValueError, TypeError):
            raise LedgerAccountingError(f"price must be a valid float, got {price}")
            
        try:
            qty = float(qty)
        except (ValueError, TypeError):
            raise LedgerAccountingError(f"qty must be a valid float, got {qty}")
            
        try:
            transaction_cost = float(transaction_cost)
        except (ValueError, TypeError):
            raise LedgerAccountingError(f"transaction_cost must be a valid float, got {transaction_cost}")
            
        if price <= 0.0:
            raise LedgerAccountingError(f"price must be > 0, got {price}")
        if qty < 0.0:
            raise LedgerAccountingError(f"qty must be >= 0, got {qty}")
        if transaction_cost < 0.0:
            raise LedgerAccountingError(f"transaction_cost must be >= 0, got {transaction_cost}")
            
        if action == "BUY" and qty <= 0.0:
            raise LedgerAccountingError("BUY qty must be > 0")
        if action in ["SELL", "CLOSE"] and qty <= 0.0:
            raise LedgerAccountingError(f"{action} qty must be > 0")
            
        # 5. Idempotency order identity sha256 checksum
        order_id = hashlib.sha256(f"{timestamp}|{self.ticker}|{action}|{price}|{qty}".encode("utf-8")).hexdigest()
        
        if order_id in self.processed_order_ids:
            logger.info(f"Order skipped: already processed with order_id {order_id}")
            return {
                "status": "IGNORED",
                "order_id": order_id,
                "timestamp": timestamp,
                "ticker": self.ticker,
                "action": action,
                "price": price,
                "qty": qty,
                "cash_before": self.cash,
                "cash_after": self.cash,
                "position_before": self.position_qty,
                "position_after": self.position_qty,
                "avg_cost": self.avg_cost,
                "realized_pnl": 0.0,
                "unrealized_pnl": self.position_qty * (price - self.avg_cost),
                "equity": self.cash + (self.position_qty * price),
                "transaction_cost": 0.0,
                "rejected_reason": None,
                "note": "Duplicate order ID skipped"
            }
            
        # 4. HOLD must not be recorded as a transaction ledger row
        if action == "HOLD":
            return {
                "status": "IGNORED",
                "order_id": order_id,
                "timestamp": timestamp,
                "ticker": self.ticker,
                "action": action,
                "price": price,
                "qty": qty,
                "cash_before": self.cash,
                "cash_after": self.cash,
                "position_before": self.position_qty,
                "position_after": self.position_qty,
                "avg_cost": self.avg_cost,
                "realized_pnl": 0.0,
                "unrealized_pnl": self.position_qty * (price - self.avg_cost),
                "equity": self.cash + (self.position_qty * price),
                "transaction_cost": transaction_cost,
                "rejected_reason": None,
                "note": "HOLD action ignored"
            }
            
        cash_before = self.cash
        pos_before = self.position_qty
        avg_cost_before = self.avg_cost
        
        cash_after = self.cash
        pos_after = self.position_qty
        avg_cost_after = self.avg_cost
        realized = 0.0
        rejected_reason = None
        note = f"Action {action} processed successfully"
        
        if action == "BUY":
            cost = qty * price
            total_cost_with_fee = cost + transaction_cost
            if total_cost_with_fee > self.cash:
                rejected_reason = "NEGATIVE_CASH_VIOLATION"
                note = f"Insufficient Cash: required={total_cost_with_fee:.2f}, balance={self.cash:.2f}"
            else:
                cash_after = self.cash - total_cost_with_fee
                pos_after = self.position_qty + qty
                # 2. BUY transaction cost must be included in avg_cost
                total_position_cost_before = pos_before * avg_cost_before
                total_new_cost = qty * price + transaction_cost
                avg_cost_after = (total_position_cost_before + total_new_cost) / pos_after
                
                self.cash = cash_after
                self.position_qty = pos_after
                self.avg_cost = avg_cost_after
                self.last_timestamp = timestamp
                
        elif action in ["SELL", "CLOSE"]:
            if qty > self.position_qty:
                rejected_reason = "NEGATIVE_POSITION_VIOLATION"
                note = f"Insufficient Shares: required={qty:.0f}, holding={self.position_qty:.0f}"
            else:
                proceeds = qty * price
                net_proceeds = proceeds - transaction_cost
                cash_after = self.cash + net_proceeds
                pos_after = self.position_qty - qty
                
                # 1. SELL realized_pnl must include transaction cost
                realized = qty * (price - avg_cost_before) - transaction_cost
                
                # 3. If SELL closes the full position, avg_cost_after must be 0.0 in the recorded ledger entry
                if pos_after <= 0.0:
                    avg_cost_after = 0.0
                else:
                    avg_cost_after = avg_cost_before
                    
                self.cash = cash_after
                self.position_qty = pos_after
                self.avg_cost = avg_cost_after
                self.last_timestamp = timestamp
                
        # 8. Rejected orders can be recorded, but must not mutate cash/position/avg_cost.
        if rejected_reason is not None:
            cash_after = cash_before
            pos_after = pos_before
            avg_cost_after = avg_cost_before
            realized = 0.0
            
        unrealized = pos_after * (price - avg_cost_after)
        equity = cash_after + (pos_after * price)
        status = "FILLED" if rejected_reason is None else "REJECTED"
        
        entry = {
            "order_id": order_id,
            "timestamp": timestamp,
            "ticker": self.ticker,
            "action": action,
            "price": price,
            "qty": qty,
            "cash_before": cash_before,
            "cash_after": cash_after,
            "position_before": pos_before,
            "position_after": pos_after,
            "avg_cost": avg_cost_after,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "equity": equity,
            "transaction_cost": transaction_cost,
            "rejected_reason": rejected_reason,
            "note": note,
            "status": status
        }
        
        self.record_entry(entry)
        self.processed_order_ids.add(order_id)
        return entry
