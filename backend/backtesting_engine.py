"""
BACKTESTING ENGINE - Validation Historique du Système de Trading Dual AI
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio

# Import our trading system components
from technical_pattern_detector import TechnicalPatternDetector
from server import TechnicalAnalysis, MarketOpportunity, TradingDecision, SignalType

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Résultats d'un backtest"""
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    trades_detail: List[Dict]

class BacktestingEngine:
    """Engine de backtesting pour valider notre système de trading"""
    
    def __init__(self):
        self.pattern_detector = TechnicalPatternDetector()
        self.initial_capital = 10000  # $10,000 capital initial
        self.risk_per_trade = 0.02  # 2% risk per trade comme notre système
        
        # Load historical data
        self.historical_data = {}
        self._load_historical_data()
        
        logger.info(f"Backtesting Engine initialized with {len(self.historical_data)} symbols")
    
    def _load_historical_data(self):
        """Charge les données historiques"""
        data_dir = "/app/historical_data"
        
        if not os.path.exists(data_dir):
            logger.error(f"Historical data directory not found: {data_dir}")
            return
        
        # Mapping des noms de fichiers vers symboles trading
        symbol_mapping = {
            'coin_Bitcoin.csv': 'BTCUSDT',
            'coin_Ethereum.csv': 'ETHUSDT', 
            'coin_BinanceCoin.csv': 'BNBUSDT',
            'coin_Cardano.csv': 'ADAUSDT',
            'coin_ChainLink.csv': 'LINKUSDT',
            'coin_Dogecoin.csv': 'DOGEUSDT',
            'coin_Polkadot.csv': 'DOTUSDT',
            'coin_USDCoin.csv': 'USDCUSDT',
            'coin_Tether.csv': 'USDTUSDT',
            'coin_Litecoin.csv': 'LTCUSDT',
            'coin_BitcoinCash.csv': 'BCHUSDT',
            'coin_Stellar.csv': 'XLMUSDT',
            'coin_Monero.csv': 'XMRUSDT',
            'coin_EOS.csv': 'EOSUSDT',
            'coin_Tron.csv': 'TRXUSDT',
            'coin_Aave.csv': 'AAVEUSDT',
            'coin_Cosmos.csv': 'ATOMUSDT'
        }
        
        for filename, symbol in symbol_mapping.items():
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    
                    # Nettoyer et formater les données
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')
                    
                    # Renommer colonnes pour correspondre à notre système
                    df = df.rename(columns={
                        'High': 'High',
                        'Low': 'Low', 
                        'Open': 'Open',
                        'Close': 'Close',
                        'Volume': 'Volume'
                    })
                    
                    # Ajouter colonnes calculées nécessaires
                    df['price_change_24h'] = df['Close'].pct_change() * 100
                    df['volume_24h'] = df['Volume']
                    
                    # Filtrer les données valides (pas de NaN sur les prix)
                    df = df.dropna(subset=['High', 'Low', 'Open', 'Close'])
                    
                    if len(df) > 100:  # Au moins 100 jours de données
                        self.historical_data[symbol] = df
                        logger.info(f"Loaded {len(df)} days of data for {symbol}")
                    else:
                        logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
                        
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        
        logger.info(f"Successfully loaded historical data for {len(self.historical_data)} symbols")
    
    async def run_comprehensive_backtest(self, start_date: str = "2020-01-01", 
                                       end_date: str = "2021-07-01",
                                       symbols: Optional[List[str]] = None) -> Dict[str, BacktestResult]:
        """Lance un backtest complet sur les données historiques"""
        
        if symbols is None:
            symbols = list(self.historical_data.keys())[:5]  # Test sur 5 symboles d'abord
        
        logger.info(f"Starting comprehensive backtest: {start_date} to {end_date}")
        logger.info(f"Testing symbols: {symbols}")
        
        results = {}
        
        for symbol in symbols:
            if symbol in self.historical_data:
                logger.info(f"Running backtest for {symbol}")
                result = await self._backtest_symbol(symbol, start_date, end_date)
                results[symbol] = result
            else:
                logger.warning(f"No historical data available for {symbol}")
        
        # Générer rapport consolidé
        self._generate_consolidated_report(results)
        
        return results
    
    async def _backtest_symbol(self, symbol: str, start_date: str, end_date: str) -> BacktestResult:
        """Backtest pour un symbole spécifique"""
        
        df = self.historical_data[symbol].copy()
        
        # Filtrer par dates
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if len(df) < 50:
            logger.warning(f"Insufficient data for {symbol} in date range")
            return self._empty_backtest_result(symbol, start_date, end_date)
        
        trades = []
        current_capital = self.initial_capital
        peak_capital = self.initial_capital
        max_drawdown = 0.0
        
        # Variables pour position ouverte
        open_position = None
        
        # Parcourir les données jour par jour
        for i in range(50, len(df)):  # Commencer à J+50 pour avoir de l'historique
            current_date = df.iloc[i]['Date']
            current_data = df.iloc[i]
            
            # Créer historique pour analyse technique (50 derniers jours)
            historical_window = df.iloc[i-49:i+1].copy()
            
            try:
                # 1. SIMULATION IA1: Analyser les patterns techniques
                patterns_detected = self.pattern_detector._detect_all_patterns(symbol, historical_window)
                
                # 2. SIMULATION IA1: Calculer indicateurs techniques
                rsi = self._calculate_rsi(historical_window['Close'])
                macd_signal = self._calculate_macd_signal(historical_window['Close'])
                
                # Créer MarketOpportunity simulée
                opportunity = MarketOpportunity(
                    symbol=symbol,
                    current_price=current_data['Close'],
                    price_change_24h=current_data['price_change_24h'],
                    volume_24h=current_data['volume_24h'],
                    market_cap=current_data.get('Marketcap', 0),
                    market_cap_rank=1
                )
                
                # Créer TechnicalAnalysis simulée
                analysis = TechnicalAnalysis(
                    symbol=symbol,
                    rsi=rsi,
                    macd_signal=macd_signal,
                    patterns_detected=[p.pattern_type.value for p in patterns_detected]
                )
                
                # 3. SIMULATION IA2: Décision de trading
                decision = await self._simulate_ia2_decision(opportunity, analysis, patterns_detected)
                
                # 4. GESTION DES POSITIONS
                if open_position is None and decision.signal != SignalType.HOLD:
                    # Ouvrir nouvelle position
                    position_size = self._calculate_position_size(
                        current_capital, decision.stop_loss, current_data['Close'], decision.signal
                    )
                    
                    if position_size > 0:
                        open_position = {
                            'symbol': symbol,
                            'entry_date': current_date,
                            'entry_price': current_data['Close'],
                            'signal': decision.signal,
                            'stop_loss': decision.stop_loss,
                            'take_profit_1': decision.take_profit_1,
                            'take_profit_2': decision.take_profit_2,
                            'position_size': position_size,
                            'confidence': decision.confidence
                        }
                
                elif open_position is not None:
                    # Vérifier conditions de sortie
                    exit_trade = self._check_exit_conditions(open_position, current_data)
                    
                    if exit_trade:
                        # Fermer position et calculer P&L
                        trade_result = self._close_position(open_position, current_data, current_date)
                        trades.append(trade_result)
                        
                        # Mettre à jour capital
                        current_capital += trade_result['pnl']
                        
                        # Calculer drawdown
                        if current_capital > peak_capital:
                            peak_capital = current_capital
                        
                        current_drawdown = (peak_capital - current_capital) / peak_capital
                        max_drawdown = max(max_drawdown, current_drawdown)
                        
                        open_position = None
                
            except Exception as e:
                logger.debug(f"Error processing {symbol} on {current_date}: {e}")
                continue
        
        # Fermer position ouverte à la fin si nécessaire
        if open_position is not None:
            final_data = df.iloc[-1]
            trade_result = self._close_position(open_position, final_data, final_data['Date'])
            trades.append(trade_result)
            current_capital += trade_result['pnl']
        
        # Calculer métriques finales
        return self._calculate_backtest_metrics(symbol, start_date, end_date, trades, 
                                              self.initial_capital, current_capital, max_drawdown)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcule RSI simplifié"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calcule MACD signal simplifié"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            return float((macd - signal).iloc[-1]) if not np.isnan((macd - signal).iloc[-1]) else 0.0
        except:
            return 0.0
    
    async def _simulate_ia2_decision(self, opportunity: MarketOpportunity, 
                                   analysis: TechnicalAnalysis, 
                                   patterns: List) -> TradingDecision:
        """Simule une décision IA2 basée sur notre logique"""
        
        # Logique simplifiée basée sur notre système
        signal = SignalType.HOLD
        confidence = 0.5
        
        # Analyse des patterns
        pattern_score = len(patterns) * 0.1  # Score basé sur nombre de patterns
        
        # Analyse technique
        technical_score = 0.0
        if analysis.rsi < 30:  # Oversold
            technical_score += 0.3
        elif analysis.rsi > 70:  # Overbought
            technical_score -= 0.3
        
        if analysis.macd_signal > 0:  # MACD bullish
            technical_score += 0.2
        elif analysis.macd_signal < 0:  # MACD bearish
            technical_score -= 0.2
        
        # Combine scores
        total_score = technical_score + pattern_score
        
        # Décision finale
        if total_score > 0.4:
            signal = SignalType.LONG
            confidence = min(0.6 + total_score, 0.95)
        elif total_score < -0.4:
            signal = SignalType.SHORT
            confidence = min(0.6 + abs(total_score), 0.95)
        else:
            signal = SignalType.HOLD
            confidence = 0.5
        
        # Calculer niveaux
        current_price = opportunity.current_price
        volatility = abs(opportunity.price_change_24h or 2.0) / 100
        
        if signal == SignalType.LONG:
            stop_loss = current_price * (1 - max(volatility * 2, 0.02))  # Min 2% SL
            take_profit_1 = current_price * (1 + max(volatility * 3, 0.03))
            take_profit_2 = current_price * (1 + max(volatility * 5, 0.05))
        elif signal == SignalType.SHORT:
            stop_loss = current_price * (1 + max(volatility * 2, 0.02))
            take_profit_1 = current_price * (1 - max(volatility * 3, 0.03))
            take_profit_2 = current_price * (1 - max(volatility * 5, 0.05))
        else:
            stop_loss = current_price * 0.98
            take_profit_1 = current_price * 1.02
            take_profit_2 = current_price * 1.02
        
        return TradingDecision(
            symbol=opportunity.symbol,
            signal=signal,
            confidence=confidence,
            entry_price=current_price,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_2,
            stop_loss=stop_loss,
            position_size=0.02,  # 2% risk
            risk_reward_ratio=abs((take_profit_1 - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 1.0,
            trailing_percentage=3.0
        )
    
    def _calculate_position_size(self, capital: float, stop_loss: float, 
                               entry_price: float, signal: SignalType) -> float:
        """Calcule la taille de position basée sur 2% risk"""
        try:
            risk_amount = capital * self.risk_per_trade  # 2% du capital
            
            if signal == SignalType.LONG:
                risk_per_share = entry_price - stop_loss
            else:  # SHORT
                risk_per_share = stop_loss - entry_price
            
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share
                # Limiter la position à 20% du capital maximum
                max_position = capital * 0.20 / entry_price
                return min(position_size, max_position)
            
            return 0.0
        except:
            return 0.0
    
    def _check_exit_conditions(self, position: Dict, current_data: pd.Series) -> bool:
        """Vérifie les conditions de sortie d'une position"""
        current_price = current_data['Close']
        
        if position['signal'] == SignalType.LONG:
            # Long position: stop loss ou take profit
            if current_price <= position['stop_loss']:
                return True
            if current_price >= position['take_profit_1']:
                return True
        
        elif position['signal'] == SignalType.SHORT:
            # Short position: stop loss ou take profit
            if current_price >= position['stop_loss']:
                return True
            if current_price <= position['take_profit_1']:
                return True
        
        return False
    
    def _close_position(self, position: Dict, exit_data: pd.Series, exit_date) -> Dict:
        """Ferme une position et calcule le P&L"""
        exit_price = exit_data['Close']
        
        if position['signal'] == SignalType.LONG:
            # Long: profit si prix monte
            pnl = (exit_price - position['entry_price']) * position['position_size']
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100
        else:
            # Short: profit si prix baisse  
            pnl = (position['entry_price'] - exit_price) * position['position_size']
            pnl_pct = (position['entry_price'] / exit_price - 1) * 100
        
        # Détecter le type de sortie
        if position['signal'] == SignalType.LONG:
            if exit_price <= position['stop_loss']:
                exit_reason = "stop_loss"
            elif exit_price >= position['take_profit_1']:
                exit_reason = "take_profit"
            else:
                exit_reason = "time_exit"
        else:
            if exit_price >= position['stop_loss']:
                exit_reason = "stop_loss"
            elif exit_price <= position['take_profit_1']:
                exit_reason = "take_profit"
            else:
                exit_reason = "time_exit"
        
        return {
            'symbol': position['symbol'],
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'signal': position['signal'],
            'position_size': position['position_size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'confidence': position['confidence'],
            'days_held': (exit_date - position['entry_date']).days if hasattr(exit_date - position['entry_date'], 'days') else 1
        }
    
    def _calculate_backtest_metrics(self, symbol: str, start_date: str, end_date: str,
                                  trades: List[Dict], initial_capital: float, 
                                  final_capital: float, max_drawdown: float) -> BacktestResult:
        """Calcule les métriques de performance du backtest"""
        
        if not trades:
            return self._empty_backtest_result(symbol, start_date, end_date)
        
        # Métriques de base
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Retours
        total_return = (final_capital - initial_capital) / initial_capital
        
        # PnL des trades
        trade_pnls = [t['pnl'] for t in trades]
        positive_pnls = [pnl for pnl in trade_pnls if pnl > 0]
        negative_pnls = [pnl for pnl in trade_pnls if pnl < 0]
        
        # Profit factor
        gross_profit = sum(positive_pnls) if positive_pnls else 0
        gross_loss = abs(sum(negative_pnls)) if negative_pnls else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # Autres métriques
        avg_trade_return = sum(trade_pnls) / total_trades if total_trades > 0 else 0
        best_trade = max(trade_pnls) if trade_pnls else 0
        worst_trade = min(trade_pnls) if trade_pnls else 0
        
        # Sharpe ratio approximatif (simplifié)
        if len(trade_pnls) > 1:
            returns_std = np.std(trade_pnls)
            sharpe_ratio = (avg_trade_return / returns_std) if returns_std != 0 else 0
        else:
            sharpe_ratio = 0
        
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades_detail=trades
        )
    
    def _empty_backtest_result(self, symbol: str, start_date: str, end_date: str) -> BacktestResult:
        """Retourne un résultat vide si pas assez de données"""
        return BacktestResult(
            symbol=symbol, start_date=start_date, end_date=end_date,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_return=0.0, max_drawdown=0.0, sharpe_ratio=0.0, profit_factor=0.0,
            avg_trade_return=0.0, best_trade=0.0, worst_trade=0.0, trades_detail=[]
        )
    
    def _generate_consolidated_report(self, results: Dict[str, BacktestResult]):
        """Génère un rapport consolidé des résultats"""
        
        logger.info("=" * 80)
        logger.info("🎯 BACKTESTING RESULTS SUMMARY")
        logger.info("=" * 80)
        
        total_trades = sum(r.total_trades for r in results.values())
        total_winning = sum(r.winning_trades for r in results.values())
        overall_win_rate = total_winning / total_trades if total_trades > 0 else 0
        
        profitable_symbols = len([r for r in results.values() if r.total_return > 0])
        
        logger.info(f"📊 OVERALL PERFORMANCE:")
        logger.info(f"   • Symbols tested: {len(results)}")
        logger.info(f"   • Profitable symbols: {profitable_symbols}/{len(results)} ({profitable_symbols/len(results)*100:.1f}%)")
        logger.info(f"   • Total trades: {total_trades}")
        logger.info(f"   • Overall win rate: {overall_win_rate:.1%}")
        
        logger.info(f"\n📈 INDIVIDUAL SYMBOL PERFORMANCE:")
        for symbol, result in results.items():
            logger.info(f"   {symbol}: {result.total_return:+.1%} return, {result.total_trades} trades, {result.win_rate:.1%} win rate")
        
        logger.info("=" * 80)

# Instance globale pour utilisation
backtesting_engine = BacktestingEngine()