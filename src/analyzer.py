# -*- coding: utf-8 -*-
"""
===================================
A股自選股智能分析系統 - AI分析層
===================================

職責：
1. 封裝 LLM 調用邏輯
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
    '600519': '貴州茅台', '000001': '平安銀行', '300750': '寧德時代', '002594': '比亞迪',
    '600036': '招商銀行', '601318': '中國平安', '000858': '五糧液', '600276': '恆瑞醫藥',
    'AAPL': '蘋果', 'TSLA': '特斯拉', 'MSFT': '微軟', 'GOOGL': '谷歌A', 'GOOG': '谷歌C',
    'AMZN': '亞馬遜', 'NVDA': '輝達', 'META': 'Meta', 'AMD': 'AMD', 'INTC': '英特爾',
    'BABA': '阿里巴巴', 'PDD': '拼多多', 'JD': '京東', 'BIDU': '百度', 'NIO': '蔚來',
    'QQQ': '納斯達克100ETF', 'DIA': '道瓊工業ETF', 'SPY': '標普500ETF', 'SOXX': '半導體ETF',
    'XLK': '科技ETF', 'XLB': '材料ETF', 'XLU': '公用事業ETF', 'XLP': '必需消費ETF', 
    'XLE': '能源ETF', 'XLF': '金融ETF', '00700': '騰訊控股', '03690': '美團'
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

    market_snapshot: Optional[Dict[str, Any]] =
