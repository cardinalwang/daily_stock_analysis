# -*- coding: utf-8 -*-
"""
===================================
A股自選股智能分析系統 - AI分析層
===================================

職責：
1. 封裝 LLM 調用邏輯（透過 LiteLLM 統一調用 Gemini/Anthropic/OpenAI 等）
2. 結合技術面和消息面生成分析報告
3. 解析 LLM 響應為結構化 AnalysisResult
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params

logger = logging.getLogger(__name__)


# 股票名稱映射（繁體化）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '貴州茅台', '000001': '平安銀行', '300750': '寧德時代', '002594': '比亞迪',
    '600036': '招商銀行', '601318': '中國平安', '000858': '五糧液', '600276': '恆瑞醫藥',
    '601012': '隆基綠能', '002475': '立訊精密', '300059': '東方財富', '002415': '海康威視',
    '600900': '長江電力', '601166': '興業銀行', '600028': '中國石化',

    # === 美股 ===
    'AAPL': '蘋果', 'TSLA': '特斯拉', 'MSFT': '微軟', 'GOOGL': '谷歌A', 'GOOG': '谷歌C',
    'AMZN': '亞馬遜', 'NVDA': '輝達', 'META': 'Meta', 'AMD': 'AMD', 'INTC': '英特爾',
    'BABA': '阿里巴巴', 'PDD': '拼多多', 'JD': '京東', 'BIDU': '百度', 'NIO': '蔚來',
    'XPEV': '小鵬汽車', 'LI': '理想汽車', 'COIN': 'Coinbase', 'MSTR': '微策略',

    # === 港股 ===
    '00700': '騰訊控股', '03690': '美團', '01810': '小米集團', '09988': '阿里巴巴',
    '09618': '京東集團', '09888': '百度集團', '01024': '快手', '00981': '中芯國際',
    '02015': '理想汽車', '09868': '小鵬汽車', '00005': '匯豐控股', '01299': '友邦保險',
    '00941': '中國移動', '00883': '中國海洋石油',
}


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """多來源獲取股票中文名稱"""
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"無法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"從數據源獲取股票名稱失敗: {e}")

    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """AI 分析結果數據類"""
    code: str
    name: str

    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"
    dashboard: Optional[Dict[str, Any]] = None

    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""

    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def get_core_conclusion(self) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            return pos_advice.get('has_position', self.operation_advice) if has_position else pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        emoji_map = {
            '買入': '🟢', '加倉': '🟢', '強烈買入': '💚', '持有': '🟡',
            '觀望': '⚪', '減倉': '🟠', '賣出': '🔴', '強烈賣出': '❌',
        }
        advice = self.operation_advice or ''
        if advice in emoji_map: return emoji_map[advice]
        
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map: return emoji_map[part]
            
        score = self.sentiment_score
        if score >= 80: return '💚'
        elif score >= 65: return '🟢'
        elif score >= 55: return '🟡'
        elif score >= 45: return '⚪'
        elif score >= 35: return '🟠'
        else: return '🔴'

    def get_confidence_stars(self) -> str:
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """Gemini AI 分析器"""

    # 遵循原邏輯，僅進行繁體化與文字精煉
    SYSTEM_PROMPT = """你是一位專注於趨勢交易的投資分析師，請生成繁體中文的【決策儀表板】報告。

## 核心交易理念

### 1. 嚴進策略（不追高）
- 乖離率 (現價 - MA5) / MA5 × 100%
- < 2%：最佳買點區間
- 2-5%：可小倉介入
- > 5%：嚴禁追高！直接判定「觀望」

### 2. 趨勢交易（順勢而為）
- 多頭排列必須條件：MA5 > MA10 > MA20
- 僅做多頭排列，空頭堅決不碰

### 3. 效率優先（籌碼結構）
- 90%籌碼集中度 < 15% 為集中
- 70-90% 獲利盤需警惕回吐
- 現價高於平均成本 5-15% 為健康

### 4. 買點偏好
- 最佳：縮量回踩 MA5 支撐
- 次優：回踩 MA10 支撐
- 跌破 MA20 則觀望

### 5. 評分標準
- 買入(60-100)：多頭排列，乖離率<5%，量能配合，無利空。
- 觀望(40-59)：乖離率>5%，均線纏繞，或有風險事件。
- 賣出(0-39)：空頭排列，跌破MA20，放量下跌或重大利空。

## 輸出格式（嚴格遵守 JSON）
```json
{
    "stock_name": "正確中文全稱",
    "sentiment_score": 0-100,
    "trend_prediction": "看多/震盪/看空",
    "operation_advice": "買入/加倉/持有/減倉/賣出/觀望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "30字以內核心結論",
            "signal_type": "🟢買入/🟡觀望/🔴賣出/⚠️風險",
            "time_sensitivity": "立即/今日內/本週內/不急",
            "position_advice": {
                "no_position": "空倉者建議",
                "has_position": "持倉者建議"
            }
        },
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均線排列狀態描述",
                "is_bullish": true/false
            },
            "price_position": {
                "current_price": 數值,
                "bias_ma5": 數值,
                "bias_status": "安全/警戒/危險",
                "support_level": 數值,
                "resistance_level": 數值
            },
            "volume_analysis": {
                "volume_status": "放量/縮量/平量",
                "volume_meaning": "量能解讀"
            },
            "chip_structure": {
                "profit_ratio": 數值,
                "chip_health": "健康/一般/警惕"
            }
        },
        "intelligence": {
            "latest_news": "近期重要新聞摘要",
            "risk_alerts": ["風險點1", "風險點2"],
            "positive_catalysts": ["利好1", "利好2"]
        },
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想買入點",
                "secondary_buy": "次優買入點",
                "stop_loss": "止損位",
                "take_profit": "目標位"
            },
            "position_strategy": {
                "suggested_position": "建議倉位"
            },
            "action_checklist": [
                "✅/⚠️/❌ 檢查項1",
                "✅/⚠️/❌ 檢查項2"
            ]
        }
    },
    "analysis_summary": "100字綜合分析摘要",
    "risk_warning": "風險提示"
}
