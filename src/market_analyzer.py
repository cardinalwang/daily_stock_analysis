# -*- coding: utf-8 -*-
"""
===================================
大盤復盤分析模組
===================================

職責：
1. 獲取大盤指數數據（上證、深證、創業板）
2. 搜索市場新聞形成復盤情報
3. 使用大模型生成每日大盤復盤報告
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """大盤指數數據"""
    code: str                    # 指數代碼
    name: str                    # 指數名稱
    current: float = 0.0         # 當前點位
    change: float = 0.0          # 漲跌點數
    change_pct: float = 0.0      # 漲跌幅(%)
    open: float = 0.0            # 開盤點位
    high: float = 0.0            # 最高點位
    low: float = 0.0             # 最低點位
    prev_close: float = 0.0      # 昨收點位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交額（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市場概覽數據"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指數
    up_count: int = 0                   # 上漲家數
    down_count: int = 0                 # 下跌家數
    flat_count: int = 0                 # 平盤家數
    limit_up_count: int = 0             # 漲停家數
    limit_down_count: int = 0           # 跌停家數
    total_amount: float = 0.0           # 兩市成交額（億元）
    # north_flow: float = 0.0           # 北向資金淨流入（億元）- 已廢棄，接口不可用
    
    # 板塊漲幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 漲幅前5板塊
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板塊


class MarketAnalyzer:
    """
    大盤復盤分析器
    
    功能：
    1. 獲取大盤指數實時行情
    2. 獲取市場漲跌統計
    3. 獲取板塊漲跌榜
    4. 搜索市場新聞
    5. 生成大盤復盤報告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
    ):
        """
        初始化大盤分析器

        Args:
            search_service: 搜索服務實例
            analyzer: AI分析器實例（用於調用LLM）
            region: 市場區域 cn=A股 us=美股
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def get_market_overview(self) -> MarketOverview:
        """
        獲取市場概覽數據
        
        Returns:
            MarketOverview: 市場概覽數據對象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 獲取主要指數行情（按 region 切換 A 股/美股）
        overview.indices = self._get_main_indices()

        # 2. 獲取漲跌統計（A 股有，美股無等效數據）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 獲取板塊漲跌榜（A 股有，美股暫無）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
        
        # 4. 獲取北向資金（可選）
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """獲取主要指數實時行情"""
        indices = []

        try:
            logger.info("[大盤] 獲取主要指數實時行情...")

            # 使用 DataFetcherManager 獲取指數行情（按 region 切換）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盤] 所有行情數據源失敗，將依賴新聞搜索進行分析")
            else:
                logger.info(f"[大盤] 獲取到 {len(indices)} 個指數行情")

        except Exception as e:
            logger.error(f"[大盤] 獲取指數行情失敗: {e}")

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """獲取市場漲跌統計"""
        try:
            logger.info("[大盤] 獲取市場漲跌統計...")

            stats = self.data_manager.get_market_stats()

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(f"[大盤] 漲:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"漲停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交額:{overview.total_amount:.0f}億")

        except Exception as e:
            logger.error(f"[大盤] 獲取漲跌統計失敗: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        """獲取板塊漲跌榜"""
        try:
            logger.info("[大盤] 獲取板塊漲跌榜...")

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(f"[大盤] 領漲板塊: {[s['name'] for s in overview.top_sectors]}")
                logger.info(f"[大盤] 領跌板塊: {[s['name'] for s in overview.bottom_sectors]}")

        except Exception as e:
            logger.error(f"[大盤] 獲取板塊漲跌榜失敗: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """獲取北向資金流入"""
    #     try:
    #         logger.info("[大盤] 獲取北向資金...")
    #         
    #         # 獲取北向資金數據
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一條數據
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 轉為億元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盤] 北向資金淨流入: {overview.north_flow:.2f}億")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盤] 獲取北向資金失敗: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市場新聞
        
        Returns:
            新聞列表
        """
        if not self.search_service:
            logger.warning("[大盤] 搜索服務未配置，跳過新聞搜索")
            return []
        
        all_news = []

        # 按 region 使用不同的新聞搜索詞
        search_queries = self.profile.news_queries
        
        try:
            logger.info("[大盤] 開始搜索市場新聞...")
            
            # 根據 region 設置搜索上下文名稱，避免美股搜索被解讀為 A 股語境
            market_name = "大盤" if self.region == "cn" else "US market"
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盤] 搜索 '{query}' 獲取 {len(response.results)} 條結果")
            
            logger.info(f"[大盤] 共獲取 {len(all_news)} 條市場新聞")
            
        except Exception as e:
            logger.error(f"[大盤] 搜索市場新聞失敗: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盤復盤報告
        
        Args:
            overview: 市場概覽數據
            news: 市場新聞列表 (SearchResult 對象列表)
            
        Returns:
            大盤復盤報告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盤] AI分析器未配置或不可用，使用模板生成報告")
            return self._generate_template_review(overview, news)
        
        # 構建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        logger.info("[大盤] 調用大模型生成復盤報告...")
        # Use the public generate_text() entry point — never access private analyzer attributes.
        review = self.analyzer.generate_text(prompt, max_tokens=2048, temperature=0.7)

        if review:
            logger.info("[大盤] 復盤報告生成成功，長度: %d 字元", len(review))
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview)
        else:
            logger.warning("[大盤] 大模型返回為空，使用模板報告")
            return self._generate_template_review(overview, news)
    
    def _inject_data_into_review(self, review: str, overview: MarketOverview) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        import re

        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)

        # Inject market stats after "### 一、市場總結" section (before next ###)
        if stats_block:
            review = self._insert_after_section(review, r'###\s*一、市場總結', stats_block)

        # Inject indices table after "### 二、指數點評" section
        if indices_block:
            review = self._insert_after_section(review, r'###\s*二、指數點評', indices_block)

        # Inject sector rankings after "### 四、熱點解讀" section
        if sector_block:
            review = self._insert_after_section(review, r'###\s*四、熱點解讀', sector_block)

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        lines = [
            f"> 📈 上漲 **{overview.up_count}** 家 / 下跌 **{overview.down_count}** 家 / "
            f"平盤 **{overview.flat_count}** 家 | "
            f"漲停 **{overview.limit_up_count}** / 跌停 **{overview.limit_down_count}** | "
            f"成交額 **{overview.total_amount:.0f}** 億"
        ]
        return "\n".join(lines)

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """構建指數行情表格（不含振幅）"""
        if not overview.indices:
            return ""
        lines = [
            "| 指數 | 最新 | 漲跌幅 | 成交額(億) |",
            "|------|------|--------|-----------|"]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_raw = idx.amount or 0.0
            if amount_raw == 0.0:
                # Yahoo Finance 不提供成交額，顯示 N/A 避免誤解
                amount_str = "N/A"
            elif amount_raw > 1e6:
                amount_str = f"{amount_raw / 1e8:.0f}"
            else:
                amount_str = f"{amount_raw:.0f}"
            lines.append(f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | {amount_str} |")
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build sector ranking block."""
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            top = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]]
            )
            lines.append(f"> 🔥 領漲: {top}")
        if overview.bottom_sectors:
            bot = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]]
            )
            lines.append(f"> 💧 領跌: {bot}")
        return "\n".join(lines)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """構建復盤報告 Prompt"""
        # 指數行情信息（簡潔格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板塊信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])
        
        # 新聞信息 - 支持 SearchResult 對象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 對象和字典
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        # 按 region 組裝市場概況與板塊區塊（美股無漲跌家數、板塊數據）
        stats_block = ""
        sector_block = ""
        if self.region == "us":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Overview
- Up: {overview.up_count} | Down: {overview.down_count} | Flat: {overview.flat_count}
- Limit up: {overview.limit_up_count} | Limit down: {overview.limit_down_count}
- Total volume (CNY bn): {overview.total_amount:.0f}"""
            else:
                stats_block = "## Market Overview\n(US market has no equivalent advance/decline stats.)"

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector Performance
Leading: {top_sectors_text if top_sectors_text else "N/A"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}"""
            else:
                sector_block = "## Sector Performance\n(US sector data not available.)"
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市場概況
- 上漲: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盤: {overview.flat_count} 家
- 漲停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 兩市成交額: {overview.total_amount:.0f} 億元"""
            else:
                stats_block = "## 市場概況\n（美股暫無漲跌家數等統計）"

            if self.profile.has_sector_rankings:
                sector_block = f"""## 板塊表現
領漲: {top_sectors_text if top_sectors_text else "暫無數據"}
領跌: {bottom_sectors_text if bottom_sectors_text else "暫無數據"}"""
            else:
                sector_block = "## 板塊表現\n（美股暫無板塊漲跌數據）"

        data_no_indices_hint = (
            "注意：由於行情數據獲取失敗，請主要根據【市場新聞】進行定性分析和總結，不要編造具體的指數點位。"
            if not indices_text
            else ""
        )
        indices_placeholder = indices_text if indices_text else ("No index data (API error)" if self.region == "us" else "暫無指數數據（接口異常）")
        news_placeholder = news_text if news_text else ("No relevant news" if self.region == "us" else "暫無相關新聞")

        # 美股場景使用英文提示語，便於生成更符合美股語境的報告
        if self.region == "us":
            data_no_indices_hint_en = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            return f"""You are a professional US/A/H market analyst. Please produce a concise US market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

## Market News
{news_placeholder}

{data_no_indices_hint_en}

{self.strategy.to_prompt_block()}

---

# Output Template (follow this structure)

## {overview.date} US Market Recap

### 1. Market Summary
(2-3 sentences on overall market performance, index moves, volume)

### 2. Index Commentary
(Analyse S&P 500, Nasdaq, Dow and other major index moves.)

### 3. Fund Flows
(Interpret volume and flow implications)

### 4. Sector/Theme Highlights
(Analyze drivers behind leading/lagging sectors)

### 5. Outlook
(Short-term view based on price action and news)

### 6. Risk Alerts
(Key risks to watch)

### 7. Strategy Plan
(Provide risk-on/neutral/risk-off stance, position sizing guideline, and one invalidation trigger.)

---

Output the report content directly, no extra commentary.
"""

        # A 股場景使用中文提示語
        return f"""你是一位專業的A/H/美股市場分析師，請根據以下數據生成一份簡潔的大盤復盤報告。

【重要】輸出要求：
- 必須輸出純 Markdown 文本格式
- 禁止輸出 JSON 格式
- 禁止輸出代碼塊
- emoji 僅在標題處少量使用（每個標題最多1個）

---

# 今日市場數據

## 日期
{overview.date}

## 主要指數
{indices_placeholder}

{stats_block}

{sector_block}

## 市場新聞
{news_placeholder}

{data_no_indices_hint}

{self.strategy.to_prompt_block()}

---

# 輸出格式模板（請嚴格按此格式輸出）

## {overview.date} 大盤復盤

### 一、市場總結
（2-3句話概括今日市場整體表現，包括指數漲跌、成交量變化）

### 二、指數點評
（{self.profile.prompt_index_hint}）

### 三、資金動向
（解讀成交額流向的含義）

### 四、熱點解讀
（分析領漲領跌板塊背後的邏輯和驅動因素）

### 五、後市展望
（結合當前走勢和新聞，給出明日市場預判）

### 六、風險提示
（需要關注的風險點）

### 七、策略計劃
（給出進攻/均衡/防守結論，對應倉位建議，並給出一個觸發失效條件；最後補充“建議僅供參考，不構成投資建議”。）

---

請直接輸出復盤報告內容，不要輸出其他說明文字。
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成復盤報告（無大模型時的備選方案）"""
        mood_code = self.profile.mood_index_code
        # 根據 mood_index_code 查找對應指數
        # cn: mood_code="000001"，idx.code 可能為 "sh000001"（以 mood_code 結尾）
        # us: mood_code="SPX"，idx.code 直接為 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = "強勢上漲"
            elif mood_index.change_pct > 0:
                market_mood = "小幅上漲"
            elif mood_index.change_pct > -1:
                market_mood = "小幅下跌"
            else:
                market_mood = "明顯下跌"
        else:
            market_mood = "震盪整理"
        
        # 指數行情（簡潔格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板塊信息
        top_text = "、".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = "、".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        # 按 region 決定是否包含漲跌統計和板塊（美股無）
        stats_section = ""
        if self.profile.has_market_stats:
            stats_section = f"""
### 三、漲跌統計
| 指標 | 數值 |
|------|------|
| 上漲家數 | {overview.up_count} |
| 下跌家數 | {overview.down_count} |
| 漲停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 兩市成交額 | {overview.total_amount:.0f}億 |
"""
        sector_section = ""
        if self.profile.has_sector_rankings and (top_text or bottom_text):
            sector_section = f"""
### 四、板塊表現
- **領漲**: {top_text}
- **領跌**: {bottom_text}
"""
        market_label = "A股" if self.region == "cn" else "美股"
        strategy_summary = self.strategy.to_markdown_block()
        report = f"""## {overview.date} 大盤復盤

### 一、市場總結
今日{market_label}市場整體呈現**{market_mood}**態勢。

### 二、主要指數
{indices_text}
{stats_section}
{sector_section}
### 五、風險提示
市場有風險，投資需謹慎。以上數據僅供參考，不構成投資建議。

{strategy_summary}

---
*復盤時間: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self) -> str:
        """
        執行每日大盤復盤流程
        
        Returns:
            復盤報告文本
        """
        logger.info("========== 開始大盤復盤分析 ==========")
        
        # 1. 獲取市場概覽
        overview = self.get_market_overview()
        
        # 2. 搜索市場新聞
        news = self.search_market_news()
        
        # 3. 生成復盤報告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盤復盤分析完成 ==========")
        
        return report


# 測試入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 測試獲取市場概覽
    overview = analyzer.get_market_overview()
    print(f"\n=== 市場概覽 ===")
    print(f"日期: {overview.date}")
    print(f"指數數量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上漲: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交額: {overview.total_amount:.0f}億")
    
    # 測試生成模板報告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 復盤報告 ===")
    print(report)
