# -*- coding: utf-8 -*-
"""
===================================
A股自選股智能分析系統 - 通知層
===================================

職責：
1. 彙整分析結果生成日報
2. 支持 Markdown 格式輸出
3. 多渠道推送（自動識別）：
   - 企業微信 Webhook
   - 飛書 Webhook
   - Telegram Bot
   - 郵件 SMTP
   - Pushover（手機/桌面推送）
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from src.config import get_config
from src.analyzer import AnalysisResult
from bot.models import BotMessage
from src.utils.data_processing import normalize_model_used
from src.notification_sender import (
    AstrbotSender,
    CustomWebhookSender,
    DiscordSender,
    EmailSender,
    FeishuSender,
    PushoverSender,
    PushplusSender,
    Serverchan3Sender,
    TelegramSender,
    WechatSender,
    WECHAT_IMAGE_MAX_BYTES
)

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """通知渠道類型"""
    WECHAT = "wechat"      # 企業微信
    FEISHU = "feishu"      # 飛書
    TELEGRAM = "telegram"  # Telegram
    EMAIL = "email"        # 郵件
    PUSHOVER = "pushover"  # Pushover（手機/桌面推送）
    PUSHPLUS = "pushplus"  # PushPlus（國內推送服務）
    SERVERCHAN3 = "serverchan3"  # Server醬3（手機APP推送服務）
    CUSTOM = "custom"      # 自定義 Webhook
    DISCORD = "discord"    # Discord 機器人 (Bot)
    ASTRBOT = "astrbot"
    UNKNOWN = "unknown"    # 未知


class ChannelDetector:
    """
    渠道檢測器 - 簡化版
    
    根據配置直接判斷渠道類型（不再需要 URL 解析）
    """
    
    @staticmethod
    def get_channel_name(channel: NotificationChannel) -> str:
        """獲取渠道中文名稱"""
        names = {
            NotificationChannel.WECHAT: "企業微信",
            NotificationChannel.FEISHU: "飛書",
            NotificationChannel.TELEGRAM: "Telegram",
            NotificationChannel.EMAIL: "郵件",
            NotificationChannel.PUSHOVER: "Pushover",
            NotificationChannel.PUSHPLUS: "PushPlus",
            NotificationChannel.SERVERCHAN3: "Server醬3",
            NotificationChannel.CUSTOM: "自定義Webhook",
            NotificationChannel.DISCORD: "Discord機器人",
            NotificationChannel.ASTRBOT: "ASTRBOT機器人",
            NotificationChannel.UNKNOWN: "未知渠道",
        }
        return names.get(channel, "未知渠道")


class NotificationService(
    AstrbotSender,
    CustomWebhookSender,
    DiscordSender,
    EmailSender,
    FeishuSender,
    PushoverSender,
    PushplusSender,
    Serverchan3Sender,
    TelegramSender,
    WechatSender
):
    """
    通知服務
    
    職責：
    1. 生成 Markdown 格式的分析日報
    2. 向所有已配置的渠道推送消息（多渠道並發）
    3. 支持本地保存日報
    
    支持的渠道：
    - 企業微信 Webhook
    - 飛書 Webhook
    - Telegram Bot
    - 郵件 SMTP
    - Pushover（手機/桌面推送）
    
    注意：所有已配置的渠道都會收到推送
    """
    
    def __init__(self, source_message: Optional[BotMessage] = None):
        """
        初始化通知服務
        
        檢測所有已配置的渠道，推送時會向所有渠道發送
        """
        config = get_config()
        self._source_message = source_message
        self._context_channels: List[str] = []

        # Markdown 轉圖片（Issue #289）
        self._markdown_to_image_channels = set(
            getattr(config, 'markdown_to_image_channels', []) or []
        )
        self._markdown_to_image_max_chars = getattr(
            config, 'markdown_to_image_max_chars', 15000
        )

        # 僅分析結果摘要（Issue #262）：true 時只推送彙整，不含個股詳情
        self._report_summary_only = getattr(config, 'report_summary_only', False)

        # 初始化各渠道
        AstrbotSender.__init__(self, config)
        CustomWebhookSender.__init__(self, config)
        DiscordSender.__init__(self, config)
        EmailSender.__init__(self, config)
        FeishuSender.__init__(self, config)
        PushoverSender.__init__(self, config)
        PushplusSender.__init__(self, config)
        Serverchan3Sender.__init__(self, config)
        TelegramSender.__init__(self, config)
        WechatSender.__init__(self, config)
        
        # 檢測所有已配置的渠道
        self._available_channels = self._detect_all_channels()
        if self._has_context_channel():
            self._context_channels.append("釘釘會話")
        
        if not self._available_channels and not self._context_channels:
            logger.warning("未配置有效的通知渠道，將不發送推送通知")
        else:
            channel_names = [ChannelDetector.get_channel_name(ch) for ch in self._available_channels]
            channel_names.extend(self._context_channels)
            logger.info(f"已配置 {len(channel_names)} 個通知渠道：{', '.join(channel_names)}")

    def _collect_models_used(self, results: List[AnalysisResult]) -> List[str]:
        models: List[str] = []
        for result in results:
            model = normalize_model_used(getattr(result, "model_used", None))
            if model:
                models.append(model)
        return list(dict.fromkeys(models))
    
    def _detect_all_channels(self) -> List[NotificationChannel]:
        """
        檢測所有已配置的渠道
        
        Returns:
            已配置的渠道列表
        """
        channels = []
        
        # 企業微信
        if self._wechat_url:
            channels.append(NotificationChannel.WECHAT)
        
        # 飛書
        if self._feishu_url:
            channels.append(NotificationChannel.FEISHU)
        
        # Telegram
        if self._is_telegram_configured():
            channels.append(NotificationChannel.TELEGRAM)
        
        # 郵件
        if self._is_email_configured():
            channels.append(NotificationChannel.EMAIL)
        
        # Pushover
        if self._is_pushover_configured():
            channels.append(NotificationChannel.PUSHOVER)

        # PushPlus
        if self._pushplus_token:
            channels.append(NotificationChannel.PUSHPLUS)

       # Server醬3
        if self._serverchan3_sendkey:
            channels.append(NotificationChannel.SERVERCHAN3)
        
        # 自定義 Webhook
        if self._custom_webhook_urls:
            channels.append(NotificationChannel.CUSTOM)
        
        # Discord
        if self._is_discord_configured():
            channels.append(NotificationChannel.DISCORD)
        # AstrBot
        if self._is_astrbot_configured():
            channels.append(NotificationChannel.ASTRBOT)
        return channels

    def is_available(self) -> bool:
        """檢查通知服務是否可用（至少有一個渠道或上下文渠道）"""
        return len(self._available_channels) > 0 or self._has_context_channel()
    
    def get_available_channels(self) -> List[NotificationChannel]:
        """獲取所有已配置的渠道"""
        return self._available_channels
    
    def get_channel_names(self) -> str:
        """獲取所有已配置渠道的名稱"""
        names = [ChannelDetector.get_channel_name(ch) for ch in self._available_channels]
        if self._has_context_channel():
            names.append("釘釘會話")
        return ', '.join(names)

    # ===== Context channel =====
    def _has_context_channel(self) -> bool:
        """判斷是否存在基於消息上下文的臨時渠道（如釘釘會話、飛書會話）"""
        return (
            self._extract_dingtalk_session_webhook() is not None
            or self._extract_feishu_reply_info() is not None
        )

    def _extract_dingtalk_session_webhook(self) -> Optional[str]:
        """從來源消息中提取釘釘會話 Webhook（用於 Stream 模式回復）"""
        if not isinstance(self._source_message, BotMessage):
            return None
        raw_data = getattr(self._source_message, "raw_data", {}) or {}
        if not isinstance(raw_data, dict):
            return None
        session_webhook = (
            raw_data.get("_session_webhook")
            or raw_data.get("sessionWebhook")
            or raw_data.get("session_webhook")
            or raw_data.get("session_webhook_url")
        )
        if not session_webhook and isinstance(raw_data.get("headers"), dict):
            session_webhook = raw_data["headers"].get("sessionWebhook")
        return session_webhook

    def _extract_feishu_reply_info(self) -> Optional[Dict[str, str]]:
        """
        從來源消息中提取飛書回復信息（用於 Stream 模式回復）
        
        Returns:
            包含 chat_id 的字典，或 None
        """
        if not isinstance(self._source_message, BotMessage):
            return None
        if getattr(self._source_message, "platform", "") != "feishu":
            return None
        chat_id = getattr(self._source_message, "chat_id", "")
        if not chat_id:
            return None
        return {"chat_id": chat_id}

    def send_to_context(self, content: str) -> bool:
        """
        向基於消息上下文的渠道發送消息（例如釘釘 Stream 會話）
        
        Args:
            content: Markdown 格式內容
        """
        return self._send_via_source_context(content)
    
    def _send_via_source_context(self, content: str) -> bool:
        """
        使用消息上下文（如釘釘/飛書會话）發送一份報告
        
        主要用於從機器人 Stream 模式觸發的任務，確保結果能回到觸發的會話。
        """
        success = False
        
        # 嘗試釘釘會話
        session_webhook = self._extract_dingtalk_session_webhook()
        if session_webhook:
            try:
                if self._send_dingtalk_chunked(session_webhook, content, max_bytes=20000):
                    logger.info("已通過釘釘會話（Stream）推送報告")
                    success = True
                else:
                    logger.error("釘釘會話（Stream）推送失敗")
            except Exception as e:
                logger.error(f"釘釘會話（Stream）推送異常: {e}")

        # 嘗試飛書會話
        feishu_info = self._extract_feishu_reply_info()
        if feishu_info:
            try:
                if self._send_feishu_stream_reply(feishu_info["chat_id"], content):
                    logger.info("已通過飛書會話（Stream）推送報告")
                    success = True
                else:
                    logger.error("飛書會話（Stream）推送失敗")
            except Exception as e:
                logger.error(f"飛書會話（Stream）推送異常: {e}")

        return success

    def _send_feishu_stream_reply(self, chat_id: str, content: str) -> bool:
        """
        通過飛書 Stream 模式發送消息到指定會話
        
        Args:
            chat_id: 飛書會話 ID
            content: 消息內容
            
        Returns:
            是否發送成功
        """
        try:
            from bot.platforms.feishu_stream import FeishuReplyClient, FEISHU_SDK_AVAILABLE
            if not FEISHU_SDK_AVAILABLE:
                logger.warning("飛書 SDK 不可用，無法發送 Stream 回復")
                return False
            
            from src.config import get_config
            config = get_config()
            
            app_id = getattr(config, 'feishu_app_id', None)
            app_secret = getattr(config, 'feishu_app_secret', None)
            
            if not app_id or not app_secret:
                logger.warning("飛書 APP_ID 或 APP_SECRET 未配置")
                return False
            
            # 創建回復客戶端
            reply_client = FeishuReplyClient(app_id, app_secret)
            
            # 飛書文本消息有長度限制，需要分批發送
            max_bytes = getattr(config, 'feishu_max_bytes', 20000)
            content_bytes = len(content.encode('utf-8'))
            
            if content_bytes > max_bytes:
                return self._send_feishu_stream_chunked(reply_client, chat_id, content, max_bytes)
            
            return reply_client.send_to_chat(chat_id, content)
            
        except ImportError as e:
            logger.error(f"導入飛書 Stream 模組失敗: {e}")
            return False
        except Exception as e:
            logger.error(f"飛書 Stream 回復異常: {e}")
            return False

    def _send_feishu_stream_chunked(
        self, 
        reply_client, 
        chat_id: str, 
        content: str, 
        max_bytes: int
    ) -> bool:
        """
        分批發送長消息到飛書（Stream 模式）
        
        Args:
            reply_client: FeishuReplyClient 實例
            chat_id: 飛書會話 ID
            content: 完整消息內容
            max_bytes: 單條消息最大字節數
            
        Returns:
            是否全部發送成功
        """
        import time
        
        def get_bytes(s: str) -> int:
            return len(s.encode('utf-8'))
        
        # 按段落或分隔線分割
        if "\n---\n" in content:
            sections = content.split("\n---\n")
            separator = "\n---\n"
        elif "\n### " in content:
            parts = content.split("\n### ")
            sections = [parts[0]] + [f"### {p}" for p in parts[1:]]
            separator = "\n"
        else:
            # 按行分割
            sections = content.split("\n")
            separator = "\n"
        
        chunks = []
        current_chunk = []
        current_bytes = 0
        separator_bytes = get_bytes(separator)
        
        for section in sections:
            section_bytes = get_bytes(section) + separator_bytes
            
            if current_bytes + section_bytes > max_bytes:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [section]
                current_bytes = section_bytes
            else:
                current_chunk.append(section)
                current_bytes += section_bytes
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        # 發送每個分塊
        success = True
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(0.5)  # 避免請求過快
            
            if not reply_client.send_to_chat(chat_id, chunk):
                success = False
                logger.error(f"飛書 Stream 分塊 {i+1}/{len(chunks)} 發送失敗")
        
        return success
        
    def generate_daily_report(
        self,
        results: List[AnalysisResult],
        report_date: Optional[str] = None
    ) -> str:
        """
        生成 Markdown 格式的日報（詳細版）

        Args:
            results: 分析結果列表
            report_date: 報告日期（默認今天）

        Returns:
            Markdown 格式的日報內容
        """
        if report_date is None:
            report_date = datetime.now().strftime('%Y-%m-%d')

        # 標題
        report_lines = [
            f"# 📅 {report_date} 股票智能分析報告",
            "",
            f"> 共分析 **{len(results)}** 隻股票 | 報告生成時間：{datetime.now().strftime('%H:%M:%S')}",
            "",
            "---",
            "",
        ]
        
        # 按評分排序（高分在前）
        sorted_results = sorted(
            results, 
            key=lambda x: x.sentiment_score, 
            reverse=True
        )
        
        # 統計資訊 - 使用 decision_type 字段準確統計
        buy_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'buy')
        sell_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'sell')
        hold_count = sum(1 for r in results if getattr(r, 'decision_type', '') in ('hold', ''))
        avg_score = sum(r.sentiment_score for r in results) / len(results) if results else 0
        
        report_lines.extend([
            "## 📊 操作建議彙整",
            "",
            "| 指標 | 數值 |",
            "|------|------|",
            f"| 🟢 建議買入/加倉 | **{buy_count}** 隻 |",
            f"| 🟡 建議持有/觀望 | **{hold_count}** 隻 |",
            f"| 🔴 建議減倉/賣出 | **{sell_count}** 隻 |",
            f"| 📈 平均看多評分 | **{avg_score:.1f}** 分 |",
            "",
            "---",
            "",
        ])
        
        # Issue #262: summary_only 時僅輸出摘要，跳過個股詳情
        if self._report_summary_only:
            report_lines.extend(["## 📊 分析結果摘要", ""])
            for r in sorted_results:
                emoji = r.get_emoji()
                report_lines.append(
                    f"{emoji} **{r.name}({r.code})**: {r.operation_advice} | "
                    f"評分 {r.sentiment_score} | {r.trend_prediction}"
                )
        else:
            report_lines.extend(["## 📈 個股詳細分析", ""])
            # 逐個股票的詳細分析
            for result in sorted_results:
                emoji = result.get_emoji()
                confidence_stars = result.get_confidence_stars() if hasattr(result, 'get_confidence_stars') else '⭐⭐'
                
                report_lines.extend([
                    f"### {emoji} {result.name} ({result.code})",
                    "",
                    f"**操作建議：{result.operation_advice}** | **綜合評分：{result.sentiment_score}分** | **趨勢預測：{result.trend_prediction}** | **置信度：{confidence_stars}**",
                    "",
                ])

                self._append_market_snapshot(report_lines, result)
                
                # 核心看點
                if hasattr(result, 'key_points') and result.key_points:
                    report_lines.extend([
                        f"**🎯 核心看點**：{result.key_points}",
                        "",
                    ])
                
                # 買入/賣出理由
                if hasattr(result, 'buy_reason') and result.buy_reason:
                    report_lines.extend([
                        f"**💡 操作理由**：{result.buy_reason}",
                        "",
                    ])
                
                # 走勢分析
                if hasattr(result, 'trend_analysis') and result.trend_analysis:
                    report_lines.extend([
                        "#### 📉 走勢分析",
                        f"{result.trend_analysis}",
                        "",
                    ])
                
                # 短期/中期展望
                outlook_lines = []
                if hasattr(result, 'short_term_outlook') and result.short_term_outlook:
                    outlook_lines.append(f"- **短期（1-3日）**：{result.short_term_outlook}")
                if hasattr(result, 'medium_term_outlook') and result.medium_term_outlook:
                    outlook_lines.append(f"- **中期（1-2週）**：{result.medium_term_outlook}")
                if outlook_lines:
                    report_lines.extend([
                        "#### 🔮 市場展望",
                        *outlook_lines,
                        "",
                    ])
                
                # 技術面分析
                tech_lines = []
                if result.technical_analysis:
                    tech_lines.append(f"**綜合**：{result.technical_analysis}")
                if hasattr(result, 'ma_analysis') and result.ma_analysis:
                    tech_lines.append(f"**均線**：{result.ma_analysis}")
                if hasattr(result, 'volume_analysis') and result.volume_analysis:
                    tech_lines.append(f"**量能**：{result.volume_analysis}")
                if hasattr(result, 'pattern_analysis') and result.pattern_analysis:
                    tech_lines.append(f"**形態**：{result.pattern_analysis}")
                if tech_lines:
                    report_lines.extend([
                        "#### 📊 技術面分析",
                        *tech_lines,
                        "",
                    ])
                
                # 基本面分析
                fund_lines = []
                if hasattr(result, 'fundamental_analysis') and result.fundamental_analysis:
                    fund_lines.append(result.fundamental_analysis)
                if hasattr(result, 'sector_position') and result.sector_position:
                    fund_lines.append(f"**板塊地位**：{result.sector_position}")
                if hasattr(result, 'company_highlights') and result.company_highlights:
                    fund_lines.append(f"**公司亮點**：{result.company_highlights}")
                if fund_lines:
                    report_lines.extend([
                        "#### 🏢 基本面分析",
                        *fund_lines,
                        "",
                    ])
                
                # 消息面/情緒面
                news_lines = []
                if result.news_summary:
                    news_lines.append(f"**新聞摘要**：{result.news_summary}")
                if hasattr(result, 'market_sentiment') and result.market_sentiment:
                    news_lines.append(f"**市場情緒**：{result.market_sentiment}")
                if hasattr(result, 'hot_topics') and result.hot_topics:
                    news_lines.append(f"**相關熱點**：{result.hot_topics}")
                if news_lines:
                    report_lines.extend([
                        "#### 📰 消息面/情緒面",
                        *news_lines,
                        "",
                    ])
                
                # 綜合分析
                if result.analysis_summary:
                    report_lines.extend([
                        "#### 📝 綜合分析",
                        result.analysis_summary,
                        "",
                    ])
                
                # 風險提示
                if hasattr(result, 'risk_warning') and result.risk_warning:
                    report_lines.extend([
                        f"⚠️ **風險提示**：{result.risk_warning}",
                        "",
                    ])
                
                # 數據來源說明
                if hasattr(result, 'search_performed') and result.search_performed:
                    report_lines.append("*🔍 已執行聯網搜索*")
                if hasattr(result, 'data_sources') and result.data_sources:
                    report_lines.append(f"*📋 數據來源：{result.data_sources}*")
                
                # 錯誤信息（如果有）
                if not result.success and result.error_message:
                    report_lines.extend([
                        "",
                        f"❌ **分析異常**：{result.error_message[:100]}",
                    ])
                
                report_lines.extend([
                    "",
                    "---",
                    "",
                ])
        
        # 底部信息（去除免責聲明）
        report_lines.extend([
            "",
            f"*報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])
        
        return "\n".join(report_lines)
    
    @staticmethod
    def _escape_md(name: str) -> str:
        """Escape markdown special characters in stock names (e.g. *ST → \\*ST)."""
        return name.replace('*', r'\*') if name else name

    @staticmethod
    def _clean_sniper_value(value: Any) -> str:
        """Normalize sniper point values and remove redundant label prefixes."""
        if value is None:
            return 'N/A'
        if isinstance(value, (int, float)):
            return str(value)
        if not isinstance(value, str):
            return str(value)
        if not value or value == 'N/A':
            return value
        prefixes = ['理想買入點：', '次優買入點：', '止損位：', '目標位：',
                    '理想買入點:', '次優買入點:', '止損位:', '目標位:']
        for prefix in prefixes:
            if value.startswith(prefix):
                return value[len(prefix):]
        return value

    def _get_signal_level(self, result: AnalysisResult) -> tuple:
        """
        Get signal level and color based on operation advice.

        Priority: advice string takes precedence over score.
        Score-based fallback is used only when advice doesn't match
        any known value.

        Returns:
            (signal_text, emoji, color_tag)
        """
        advice = result.operation_advice
        score = result.sentiment_score

        # Advice-first lookup (exact match takes priority)
        advice_map = {
            '強烈買入': ('強烈買入', '💚', '強買'),
            '買入': ('買入', '🟢', '買入'),
            '加倉': ('買入', '🟢', '買入'),
            '持有': ('持有', '🟡', '持有'),
            '觀望': ('觀望', '⚪', '觀望'),
            '減倉': ('減倉', '🟠', '減倉'),
            '賣出': ('賣出', '🔴', '賣出'),
            '強烈賣出': ('賣出', '🔴', '賣出'),
        }
        if advice in advice_map:
            return advice_map[advice]

        # Score-based fallback when advice is unrecognized
        if score >= 80:
            return ('強烈買入', '💚', '強買')
        elif score >= 65:
            return ('買入', '🟢', '買入')
        elif score >= 55:
            return ('持有', '🟡', '持有')
        elif score >= 45:
            return ('觀望', '⚪', '觀望')
        elif score >= 35:
            return ('減倉', '🟠', '減倉')
        elif score < 35:
            return ('賣出', '🔴', '賣出')
        else:
            return ('觀望', '⚪', '觀望')
    
    def generate_dashboard_report(
        self,
        results: List[AnalysisResult],
        report_date: Optional[str] = None
    ) -> str:
        """
        生成決策儀表板格式的日報（詳細版）

        格式：市場概覽 + 重要資訊 + 核心結論 + 數據透視 + 作戰計畫

        Args:
            results: 分析結果列表
            report_date: 報告日期（默認今天）

        Returns:
            Markdown 格式的決策儀表板日報
        """
        if report_date is None:
            report_date = datetime.now().strftime('%Y-%m-%d')

        # 按評分排序（高分在前）
        sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)

        # 統計資訊 - 使用 decision_type 字段準確統計
        buy_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'buy')
        sell_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'sell')
        hold_count = sum(1 for r in results if getattr(r, 'decision_type', '') in ('hold', ''))

        report_lines = [
            f"# 🎯 {report_date} 決策儀表板",
            "",
            f"> 共分析 **{len(results)}** 隻股票 | 🟢買入:{buy_count} 🟡觀望:{hold_count} 🔴賣出:{sell_count}",
            "",
        ]

        # === 新增：分析結果摘要 (Issue #112) ===
        if results:
            report_lines.extend([
                "## 📊 分析結果摘要",
                "",
            ])
            for r in sorted_results:
                _, signal_emoji, _ = self._get_signal_level(r)
                display_name = self._escape_md(r.name)
                report_lines.append(
                    f"{signal_emoji} **{display_name}({r.code})**: {r.operation_advice} | "
                    f"評分 {r.sentiment_score} | {r.trend_prediction}"
                )
            report_lines.extend([
                "",
                "---",
                "",
            ])

        # 逐個股票的決策儀表板（Issue #262: summary_only 時跳過詳情）
        if not self._report_summary_only:
            for result in sorted_results:
                signal_text, signal_emoji, signal_tag = self._get_signal_level(result)
                dashboard = result.dashboard if hasattr(result, 'dashboard') and result.dashboard else {}
                
                # 股票名稱（優先使用 dashboard 或 result 中的名稱，轉義 *ST 等特殊字符）
                raw_name = result.name if result.name and not result.name.startswith('股票') else f'股票{result.code}'
                stock_name = self._escape_md(raw_name)
                
                report_lines.extend([
                    f"## {signal_emoji} {stock_name} ({result.code})",
                    "",
                ])
                
                # ========== 輿情與基本面概覽（放在最前面）==========
                intel = dashboard.get('intelligence', {}) if dashboard else {}
                if intel:
                    report_lines.extend([
                        "### 📰 重要資訊速覽",
                        "",
                    ])
                    # 輿情情緒總結
                    if intel.get('sentiment_summary'):
                        report_lines.append(f"**💭 輿情情緒**: {intel['sentiment_summary']}")
                    # 業績預期
                    if intel.get('earnings_outlook'):
                        report_lines.append(f"**📊 業績預期**: {intel['earnings_outlook']}")
                    # 風險警報（醒目顯示）
                    risk_alerts = intel.get('risk_alerts', [])
                    if risk_alerts:
                        report_lines.append("")
                        report_lines.append("**🚨 風險警報**:")
                        for alert in risk_alerts:
                            report_lines.append(f"- {alert}")
                    # 利多催化
                    catalysts = intel.get('positive_catalysts', [])
                    if catalysts:
                        report_lines.append("")
                        report_lines.append("**✨ 利多催化**:")
                        for cat in catalysts:
                            report_lines.append(f"- {cat}")
                    # 最新動態
                    if intel.get('latest_news'):
                        report_lines.append("")
                        report_lines.append(f"**📢 最新動態**: {intel['latest_news']}")
                    report_lines.append("")
                
                # ========== 核心結論 ==========
                core = dashboard.get('core_conclusion', {}) if dashboard else {}
                one_sentence = core.get('one_sentence', result.analysis_summary)
                time_sense = core.get('time_sensitivity', '本週內')
                pos_advice = core.get('position_advice', {})
                
                report_lines.extend([
                    "### 📌 核心結論",
                    "",
                    f"**{signal_emoji} {signal_text}** | {result.trend_prediction}",
                    "",
                    f"> **一句話決策**: {one_sentence}",
                    "",
                    f"⏰ **時效性**: {time_sense}",
                    "",
                ])
                # 持倉分類建議
                if pos_advice:
                    report_lines.extend([
                        "| 持倉情況 | 操作建議 |",
                        "|---------|---------|",
                        f"| 🆕 **空倉者** | {pos_advice.get('no_position', result.operation_advice)} |",
                        f"| 💼 **持倉者** | {pos_advice.get('has_position', '繼續持有')} |",
                        "",
                    ])

                self._append_market_snapshot(report_lines, result)
                
                # ========== 數據透視 ==========
                data_persp = dashboard.get('data_perspective', {}) if dashboard else {}
                if data_persp:
                    trend_data = data_persp.get('trend_status', {})
                    price_data = data_persp.get('price_position', {})
                    vol_data = data_persp.get('volume_analysis', {})
                    chip_data = data_persp.get('chip_structure', {})
                    
                    report_lines.extend([
                        "### 📊 數據透視",
                        "",
                    ])
                    # 趨勢狀態
                    if trend_data:
                        is_bullish = "✅ 是" if trend_data.get('is_bullish', False) else "❌ 否"
                        report_lines.extend([
                            f"**均線排列**: {trend_data.get('ma_alignment', 'N/A')} | 多頭排列: {is_bullish} | 趨勢強度: {trend_data.get('trend_score', 'N/A')}/100",
                            "",
                        ])
                    # 價格位置
                    if price_data:
                        bias_status = price_data.get('bias_status', 'N/A')
                        bias_emoji = "✅" if bias_status == "安全" else ("⚠️" if bias_status == "警戒" else "🚨")
                        report_lines.extend([
                            "| 價格指標 | 數值 |",
                            "|---------|------|",
                            f"| 當前價 | {price_data.get('current_price', 'N/A')} |",
                            f"| MA5 | {price_data.get('ma5', 'N/A')} |",
                            f"| MA10 | {price_data.get('ma10', 'N/A')} |",
                            f"| MA20 | {price_data.get('ma20', 'N/A')} |",
                            f"| 乖離率(MA5) | {price_data.get('bias_ma5', 'N/A')}% {bias_emoji}{bias_status} |",
                            f"| 支撐位 | {price_data.get('support_level', 'N/A')} |",
                            f"| 壓力位 | {price_data.get('resistance_level', 'N/A')} |",
                            "",
                        ])
                    # 量能分析
                    if vol_data:
                        report_lines.extend([
                            f"**量能**: 量比 {vol_data.get('volume_ratio', 'N/A')} ({vol_data.get('volume_status', '')}) | 換手率 {vol_data.get('turnover_rate', 'N/A')}%",
                            f"💡 *{vol_data.get('volume_meaning', '')}*",
                            "",
                        ])
                    # 籌碼結構
                    if chip_data:
                        chip_health = chip_data.get('chip_health', 'N/A')
                        chip_emoji = "✅" if chip_health == "健康" else ("⚠️" if chip_health == "一般" else "🚨")
                        report_lines.extend([
                            f"**籌碼**: 獲利比例 {chip_data.get('profit_ratio', 'N/A')} | 平均成本 {chip_data.get('avg_cost', 'N/A')} | 集中度 {chip_data.get('concentration', 'N/A')} {chip_emoji}{chip_health}",
                            "",
                        ])
                
                # ========== 作戰計畫 ==========
                battle = dashboard.get('battle_plan', {}) if dashboard else {}
                if battle:
                    report_lines.extend([
                        "### 🎯 作戰計畫",
                        "",
                    ])
                    # 狙擊點位
                    sniper = battle.get('sniper_points', {})
                    if sniper:
                        report_lines.extend([
                            "**📍 狙擊點位**",
                            "",
                            "| 點位類型 | 價格 |",
                            "|---------|------|",
                            f"| 🎯 理想買入點 | {self._clean_sniper_value(sniper.get('ideal_buy', 'N/A'))} |",
                            f"| 🔵 次優買入點 | {self._clean_sniper_value(sniper.get('secondary_buy', 'N/A'))} |",
                            f"| 🛑 止損位 | {self._clean_sniper_value(sniper.get('stop_loss', 'N/A'))} |",
                            f"| 🎊 目標位 | {self._clean_sniper_value(sniper.get('take_profit', 'N/A'))} |",
                            "",
                        ])
                    # 倉位策略
                    position = battle.get('position_strategy', {})
                    if position:
                        report_lines.extend([
                            f"**💰 倉位建議**: {position.get('suggested_position', 'N/A')}",
                            f"- 建倉策略: {position.get('entry_plan', 'N/A')}",
                            f"- 風控策略: {position.get('risk_control', 'N/A')}",
                            "",
                        ])
                    # 檢查清單
                    checklist = battle.get('action_checklist', []) if battle else []
                    if checklist:
                        report_lines.extend([
                            "**✅ 檢查清單**",
                            "",
                        ])
                        for item in checklist:
                            report_lines.append(f"- {item}")
                        report_lines.append("")
                
                # 如果沒有 dashboard，顯示傳統格式
                if not dashboard:
                    # 操作理由
                    if result.buy_reason:
                        report_lines.extend([
                            f"**💡 操作理由**: {result.buy_reason}",
                            "",
                        ])
                    # 風險提示
                    if result.risk_warning:
                        report_lines.extend([
                            f"**⚠️ 風險提示**: {result.risk_warning}",
                            "",
                        ])
                    # 技術面分析
                    if result.ma_analysis or result.volume_analysis:
                        report_lines.extend([
                            "### 📊 技術面",
                            "",
                        ])
                        if result.ma_analysis:
                            report_lines.append(f"**均線**: {result.ma_analysis}")
                        if result.volume_analysis:
                            report_lines.append(f"**量能**: {result.volume_analysis}")
                        report_lines.append("")
                    # 消息面
                    if result.news_summary:
                        report_lines.extend([
                            "### 📰 消息面",
                            f"{result.news_summary}",
                            "",
                        ])
                
                report_lines.extend([
                    "---",
                    "",
                ])
        
        # 底部
        report_lines.extend([
            "",
            f"*報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])
        
        return "\n".join(report_lines)
    
    def generate_wechat_dashboard(self, results: List[AnalysisResult]) -> str:
        """
        生成企業微信決策儀表板精簡版（控制在4000字符內）
        
        只保留核心結論和狙擊點位
        
        Args:
            results: 分析結果列表
            
        Returns:
            精簡版決策儀表板
        """
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        # 按評分排序
        sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
        
        # 統計 - 使用 decision_type 字段準確統計
        buy_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'buy')
        sell_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'sell')
        hold_count = sum(1 for r in results if getattr(r, 'decision_type', '') in ('hold', ''))
        
        lines = [
            f"## 🎯 {report_date} 決策儀表板",
            "",
            f"> {len(results)}隻股票 | 🟢買入:{buy_count} 🟡觀望:{hold_count} 🔴賣出:{sell_count}",
            "",
        ]
        
        # Issue #262: summary_only 時僅輸出摘要列表
        if self._report_summary_only:
            lines.append("**📊 分析結果摘要**")
            lines.append("")
            for r in sorted_results:
                _, signal_emoji, _ = self._get_signal_level(r)
                stock_name = self._escape_md(r.name if r.name and not r.name.startswith('股票') else f'股票{r.code}')
                lines.append(
                    f"{signal_emoji} **{stock_name}({r.code})**: {r.operation_advice} | "
                    f"評分 {r.sentiment_score} | {r.trend_prediction}"
                )
        else:
            for result in sorted_results:
                signal_text, signal_emoji, _ = self._get_signal_level(result)
                dashboard = result.dashboard if hasattr(result, 'dashboard') and result.dashboard else {}
                core = dashboard.get('core_conclusion', {}) if dashboard else {}
                battle = dashboard.get('battle_plan', {}) if dashboard else {}
                intel = dashboard.get('intelligence', {}) if dashboard else {}
                
                # 股票名稱
                stock_name = result.name if result.name and not result.name.startswith('股票') else f'股票{result.code}'
                stock_name = self._escape_md(stock_name)
                
                # 標題行：信號等級 + 股票名稱
                lines.append(f"### {signal_emoji} **{signal_text}** | {stock_name}({result.code})")
                lines.append("")
                
                # 核心決策（一句話）
                one_sentence = core.get('one_sentence', result.analysis_summary) if core else result.analysis_summary
                if one_sentence:
                    lines.append(f"📌 **{one_sentence[:80]}**")
                    lines.append("")
                
                # 重要資訊區（輿情+基本面）
                info_lines = []
                
                # 業績預期
                if intel.get('earnings_outlook'):
                    outlook = intel['earnings_outlook'][:60]
                    info_lines.append(f"📊 業績: {outlook}")
                if intel.get('sentiment_summary'):
                    sentiment = intel['sentiment_summary'][:50]
                    info_lines.append(f"💭 輿情: {sentiment}")
                if info_lines:
                    lines.extend(info_lines)
                    lines.append("")
                
                # 風險警報（最重要，醒目顯示）
                risks = intel.get('risk_alerts', []) if intel else []
                if risks:
                    lines.append("🚨 **風險**:")
                    for risk in risks[:2]:  # 最多顯示2條
                        risk_text = risk[:50] + "..." if len(risk) > 50 else risk
                        lines.append(f"   • {risk_text}")
                    lines.append("")
                
                # 利多催化
                catalysts = intel.get('positive_catalysts', []) if intel else []
                if catalysts:
                    lines.append("✨ **利多**:")
                    for cat in catalysts[:2]:  # 最多顯示2條
                        cat_text = cat[:50] + "..." if len(cat) > 50 else cat
                        lines.append(f"   • {cat_text}")
                    lines.append("")
                
                # 狙擊點位
                sniper = battle.get('sniper_points', {}) if battle else {}
                if sniper:
                    ideal_buy = sniper.get('ideal_buy', '')
                    stop_loss = sniper.get('stop_loss', '')
                    take_profit = sniper.get('take_profit', '')
                    points = []
                    if ideal_buy:
                        points.append(f"🎯買點:{ideal_buy[:15]}")
                    if stop_loss:
                        points.append(f"🛑止損:{stop_loss[:15]}")
                    if take_profit:
                        points.append(f"🎊目標:{take_profit[:15]}")
                    if points:
                        lines.append(" | ".join(points))
                        lines.append("")
                
                # 持倉建議
                pos_advice = core.get('position_advice', {}) if core else {}
                if pos_advice:
                    no_pos = pos_advice.get('no_position', '')
                    has_pos = pos_advice.get('has_position', '')
                    if no_pos:
                        lines.append(f"🆕 空倉者: {no_pos[:50]}")
                    if has_pos:
                        lines.append(f"💼 持倉者: {has_pos[:50]}")
                    lines.append("")
                
                # 檢查清單簡化版
                checklist = battle.get('action_checklist', []) if battle else []
                if checklist:
                    # 只顯示不通過的項目
                    failed_checks = [c for c in checklist if c.startswith('❌') or c.startswith('⚠️')]
                    if failed_checks:
                        lines.append("**檢查未通過項**:")
                        for check in failed_checks[:3]:
                            lines.append(f"   {check[:40]}")
                        lines.append("")
                
                lines.append("---")
                lines.append("")
        
        # 底部
        lines.append(f"*生成時間: {datetime.now().strftime('%H:%M')}*")
        models = self._collect_models_used(results)
        if models:
            lines.append(f"*分析模型: {', '.join(models)}*")

        content = "\n".join(lines)
        
        return content
    
    def generate_wechat_summary(self, results: List[AnalysisResult]) -> str:
        """
        生成企業微信精簡版日報（控制在4000字符內）

        Args:
            results: 分析結果列表

        Returns:
            精簡版 Markdown 內容
        """
        report_date = datetime.now().strftime('%Y-%m-%d')

        # 按評分排序
        sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)

        # 統計 - 使用 decision_type 字段準確統計
        buy_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'buy')
        sell_count = sum(1 for r in results if getattr(r, 'decision_type', '') == 'sell')
        hold_count = sum(1 for r in results if getattr(r, 'decision_type', '') in ('hold', ''))
        avg_score = sum(r.sentiment_score for r in results) / len(results) if results else 0

        lines = [
            f"## 📅 {report_date} 股票分析報告",
            "",
            f"> 共 **{len(results)}** 隻 | 🟢買入:{buy_count} 🟡持有:{hold_count} 🔴賣出:{sell_count} | 均分:{avg_score:.0f}",
            "",
        ]
        
        # 每隻股票精簡資訊（控制長度）
        for result in sorted_results:
            emoji = result.get_emoji()
            
            # 核心資訊行
            lines.append(f"### {emoji} {result.name}({result.code})")
            lines.append(f"**{result.operation_advice}** | 評分:{result.sentiment_score} | {result.trend_prediction}")
            
            # 操作理由（截斷）
            if hasattr(result, 'buy_reason') and result.buy_reason:
                reason = result.buy_reason[:80] + "..." if len(result.buy_reason) > 80 else result.buy_reason
                lines.append(f"💡 {reason}")
            
            # 核心看點
            if hasattr(result, 'key_points') and result.key_points:
                points = result.key_points[:60] + "..." if len(result.key_points) > 60 else result.key_points
                lines.append(f"🎯 {points}")
            
            # 風險提示（截斷）
            if hasattr(result, 'risk_warning') and result.risk_warning:
                risk = result.risk_warning[:50] + "..." if len(result.risk_warning) > 50 else result.risk_warning
                lines.append(f"⚠️ {risk}")
            
            lines.append("")
        
        # 底部（模型行在 --- 之前，Issue #528）
        models = self._collect_models_used(results)
        if models:
            lines.append(f"*分析模型: {', '.join(models)}*")
        lines.extend([
            "---",
            "*AI生成，僅供參考，不構成投資建議*",
            f"*詳細報告見 reports/report_{report_date.replace('-', '')}.md*"
        ])

        content = "\n".join(lines)

        return content

    def generate_single_stock_report(self, result: AnalysisResult) -> str:
        """
        生成單隻股票的分析報告（用於單股推送模式 #55）
        
        格式精簡但資訊完整，適合每分析完一隻股票立即推送
        
        Args:
            result: 單隻股票的分析結果
            
        Returns:
            Markdown 格式的單股報告
        """
        report_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        signal_text, signal_emoji, _ = self._get_signal_level(result)
        dashboard = result.dashboard if hasattr(result, 'dashboard') and result.dashboard else {}
        core = dashboard.get('core_conclusion', {}) if dashboard else {}
        battle = dashboard.get('battle_plan', {}) if dashboard else {}
        intel = dashboard.get('intelligence', {}) if dashboard else {}
        
        # 股票名稱（轉義 *ST 等特殊字符）
        raw_name = result.name if result.name and not result.name.startswith('股票') else f'股票{result.code}'
        stock_name = self._escape_md(raw_name)
        
        lines = [
            f"## {signal_emoji} {stock_name} ({result.code})",
            "",
            f"> {report_date} | 評分: **{result.sentiment_score}** | {result.trend_prediction}",
            "",
        ]

        self._append_market_snapshot(lines, result)
        
        # 核心決策（一句話）
        one_sentence = core.get('one_sentence', result.analysis_summary) if core else result.analysis_summary
        if one_sentence:
            lines.extend([
                "### 📌 核心結論",
                "",
                f"**{signal_text}**: {one_sentence}",
                "",
            ])
        
        # 重要資訊（輿情+基本面）
        info_added = False
        if intel:
            if intel.get('earnings_outlook'):
                if not info_added:
                    lines.append("### 📰 重要資訊")
                    lines.append("")
                    info_added = True
                lines.append(f"📊 **業績預期**: {intel['earnings_outlook'][:100]}")
            
            if intel.get('sentiment_summary'):
                if not info_added:
                    lines.append("### 📰 重要資訊")
                    lines.append("")
                    info_added = True
                lines.append(f"💭 **輿情情緒**: {intel['sentiment_summary'][:80]}")
            
            # 風險警報
            risks = intel.get('risk_alerts', [])
            if risks:
                if not info_added:
                    lines.append("### 📰 重要資訊")
                    lines.append("")
                    info_added = True
                lines.append("")
                lines.append("🚨 **風險警報**:")
                for risk in risks[:3]:
                    lines.append(f"- {risk[:60]}")
            
            # 利多催化
            catalysts = intel.get('positive_catalysts', [])
            if catalysts:
                lines.append("")
                lines.append("✨ **利多催化**:")
                for cat in catalysts[:3]:
                    lines.append(f"- {cat[:60]}")
        
        if info_added:
            lines.append("")
        
        # 狙擊點位
        sniper = battle.get('sniper_points', {}) if battle else {}
        if sniper:
            lines.extend([
                "### 🎯 操作點位",
                "",
                "| 買點 | 止損 | 目標 |",
                "|------|------|------|",
            ])
            ideal_buy = sniper.get('ideal_buy', '-')
            stop_loss = sniper.get('stop_loss', '-')
            take_profit = sniper.get('take_profit', '-')
            lines.append(f"| {ideal_buy} | {stop_loss} | {take_profit} |")
            lines.append("")
        
        # 持倉建議
        pos_advice = core.get('position_advice', {}) if core else {}
        if pos_advice:
            lines.extend([
                "### 💼 持倉建議",
                "",
                f"- 🆕 **空倉者**: {pos_advice.get('no_position', result.operation_advice)}",
                f"- 💼 **持倉者**: {pos_advice.get('has_position', '繼續持有')}",
                "",
            ])
        
        lines.append("---")
        model_used = normalize_model_used(getattr(result, "model_used", None))
        if model_used:
            lines.append(f"*分析模型: {model_used}*")
        lines.append("*AI生成，僅供參考，不構成投資建議*")

        return "\n".join(lines)

    # Display name mapping for realtime data sources
    _SOURCE_DISPLAY_NAMES = {
        "tencent": "騰訊財經",
        "akshare_em": "東方財富",
        "akshare_sina": "新浪財經",
        "akshare_qq": "騰訊財經",
        "efinance": "東方財富(efinance)",
        "tushare": "Tushare Pro",
        "sina": "新浪財經",
        "fallback": "降級兜底",
    }

    def _append_market_snapshot(self, lines: List[str], result: AnalysisResult) -> None:
        snapshot = getattr(result, 'market_snapshot', None)
        if not snapshot:
            return

        lines.extend([
            "### 📈 當日行情",
            "",
            "| 收盤 | 昨收 | 開盤 | 最高 | 最低 | 漲跌幅 | 漲跌額 | 振幅 | 成交量 | 成交額 |",
            "|------|------|------|------|------|-------|-------|------|--------|--------|",
            f"| {snapshot.get('close', 'N/A')} | {snapshot.get('prev_close', 'N/A')} | "
            f"{snapshot.get('open', 'N/A')} | {snapshot.get('high', 'N/A')} | "
            f"{snapshot.get('low', 'N/A')} | {snapshot.get('pct_chg', 'N/A')} | "
            f"{snapshot.get('change_amount', 'N/A')} | {snapshot.get('amplitude', 'N/A')} | "
            f"{snapshot.get('volume', 'N/A')} | {snapshot.get('amount', 'N/A')} |",
        ])

        if "price" in snapshot:
            raw_source = snapshot.get('source', 'N/A')
            display_source = self._SOURCE_DISPLAY_NAMES.get(raw_source, raw_source)
            lines.extend([
                "",
                "| 當前價 | 量比 | 換手率 | 行情來源 |",
                "|-------|------|--------|----------|",
                f"| {snapshot.get('price', 'N/A')} | {snapshot.get('volume_ratio', 'N/A')} | "
                f"{snapshot.get('turnover_rate', 'N/A')} | {display_source} |",
            ])

        lines.append("")

    def _should_use_image_for_channel(
        self, channel: NotificationChannel, image_bytes: Optional[bytes]
    ) -> bool:
        """
        Decide whether to send as image for the given channel (Issue #289).

        Fallback rules (send as Markdown text instead of image):
        - image_bytes is None: conversion failed / imgkit not installed / content over max_chars
        - WeChat: image exceeds ~2MB limit
        """
        if channel.value not in self._markdown_to_image_channels or image_bytes is None:
            return False
        if channel == NotificationChannel.WECHAT and len(image_bytes) > WECHAT_IMAGE_MAX_BYTES:
            logger.warning(
                "企業微信圖片超限 (%d bytes)，回退為 Markdown 文本發送",
                len(image_bytes),
            )
            return False
        return True

    def send(
        self,
        content: str,
        email_stock_codes: Optional[List[str]] = None,
        email_send_to_all: bool = False
    ) -> bool:
        """
        統一發送接口 - 向所有已配置的渠道發送

        遍歷所有已配置的渠道，逐一發送消息

        Fallback rules (Markdown-to-image, Issue #289):
        - When image_bytes is None (conversion failed / imgkit not installed /
          content over max_chars): all channels configured for image will send
          as Markdown text instead.
        - When WeChat image exceeds ~2MB: that channel falls back to Markdown text.

        Args:
            content: 消息內容（Markdown 格式）
            email_stock_codes: 股票代碼列表（可選，用於郵件渠道路由到對應分組信箱，Issue #268）
            email_send_to_all: 郵件是否發往所有配置信箱（用於大盤復盤等無股票歸屬的內容）

        Returns:
            是否至少有一個渠道發送成功
        """
        context_success = self.send_to_context(content)

        if not self._available_channels:
            if context_success:
                logger.info("已通過消息上下文渠道完成推送（無其他通知渠道）")
                return True
            logger.warning("通知服務不可用，跳過推送")
            return False

        # Markdown to image (Issue #289): convert once if any channel needs it.
        # Per-channel decision via _should_use_image_for_channel (see send() docstring for fallback rules).
        image_bytes = None
        channels_needing_image = {
            ch for ch in self._available_channels
            if ch.value in self._markdown_to_image_channels
        }
        if channels_needing_image:
            from src.md2img import markdown_to_image
            image_bytes = markdown_to_image(
                content, max_chars=self._markdown_to_image_max_chars
            )
            if image_bytes:
                logger.info("Markdown 已轉換為圖片，將向 %s 發送圖片",
                            [ch.value for ch in channels_needing_image])
            elif channels_needing_image:
                try:
                    from src.config import get_config
                    engine = getattr(get_config(), "md2img_engine", "wkhtmltoimage")
                except Exception:
                    engine = "wkhtmltoimage"
                hint = (
                    "npm i -g markdown-to-file" if engine == "markdown-to-file"
                    else "wkhtmltopdf (apt install wkhtmltopdf / brew install wkhtmltopdf)"
                )
                logger.warning(
                    "Markdown 轉圖片失敗，將回退為文本發送。請檢查 MARKDOWN_TO_IMAGE_CHANNELS 配置並安裝 %s",
                    hint,
                )

        channel_names = self.get_channel_names()
        logger.info(f"正在向 {len(self._available_channels)} 個渠道發送通知：{channel_names}")

        success_count = 0
        fail_count = 0

        for channel in self._available_channels:
            channel_name = ChannelDetector.get_channel_name(channel)
            use_image = self._should_use_image_for_channel(channel, image_bytes)
            try:
                if channel == NotificationChannel.WECHAT:
                    if use_image:
                        result = self._send_wechat_image(image_bytes)
                    else:
                        result = self.send_to_wechat(content)
                elif channel == NotificationChannel.FEISHU:
                    result = self.send_to_feishu(content)
                elif channel == NotificationChannel.TELEGRAM:
                    if use_image:
                        result = self._send_telegram_photo(image_bytes)
                    else:
                        result = self.send_to_telegram(content)
                elif channel == NotificationChannel.EMAIL:
                    receivers = None
                    if email_send_to_all and self._stock_email_groups:
                        receivers = self.get_all_email_receivers()
                    elif email_stock_codes and self._stock_email_groups:
                        receivers = self.get_receivers_for_stocks(email_stock_codes)
                    if use_image:
                        result = self._send_email_with_inline_image(
                            image_bytes, receivers=receivers
                        )
                    else:
                        result = self.send_to_email(content, receivers=receivers)
                elif channel == NotificationChannel.PUSHOVER:
                    result = self.send_to_pushover(content)
                elif channel == NotificationChannel.PUSHPLUS:
                    result = self.send_to_pushplus(content)
                elif channel == NotificationChannel.SERVERCHAN3:
                    result = self.send_to_serverchan3(content)
                elif channel == NotificationChannel.CUSTOM:
                    if use_image:
                        result = self._send_custom_webhook_image(
                            image_bytes, fallback_content=content
                        )
                    else:
                        result = self.send_to_custom(content)
                elif channel == NotificationChannel.DISCORD:
                    result = self.send_to_discord(content)
                elif channel == NotificationChannel.ASTRBOT:
                    result = self.send_to_astrbot(content)
                else:
                    logger.warning(f"不支持的通知渠道: {channel}")
                    result = False

                if result:
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                logger.error(f"{channel_name} 發送失敗: {e}")
                fail_count += 1

        logger.info(f"通知發送完成：成功 {success_count} 個，失敗 {fail_count} 個")
        return success_count > 0 or context_success
    
    def save_report_to_file(
        self, 
        content: str, 
        filename: Optional[str] = None
    ) -> str:
        """
        保存日報到本地文件
        
        Args:
            content: 日報內容
            filename: 文件名（可選，默認按日期生成）
            
        Returns:
            保存的文件路徑
        """
        from pathlib import Path
        
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"report_{date_str}.md"
        
        # 確保 reports 目錄存在（使用項目根目錄下的 reports）
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"日報已保存到: {filepath}")
        return str(filepath)


class NotificationBuilder:
    """
    通知消息構建器
    
    提供便捷的消息構建方法
    """
    
    @staticmethod
    def build_simple_alert(
        title: str,
        content: str,
        alert_type: str = "info"
    ) -> str:
        """
        構建簡單的提醒消息
        
        Args:
            title: 標題
            content: 內容
            alert_type: 類型（info, warning, error, success）
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
        }
        emoji = emoji_map.get(alert_type, "📢")
        
        return f"{emoji} **{title}**\n\n{content}"
    
    @staticmethod
    def build_stock_summary(results: List[AnalysisResult]) -> str:
        """
        構建股票摘要（簡短版）
        
        適用於快速通知
        """
        lines = ["📊 **今日自選股摘要**", ""]
        
        for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
            emoji = r.get_emoji()
            lines.append(f"{emoji} {r.name}({r.code}): {r.operation_advice} | 評分 {r.sentiment_score}")
        
        return "\n".join(lines)


# 便捷函數
def get_notification_service() -> NotificationService:
    """獲取通知服務實例"""
    return NotificationService()


def send_daily_report(results: List[AnalysisResult]) -> bool:
    """
    發送每日報告的快捷方式
    
    自動識別渠道並推送
    """
    service = get_notification_service()
    
    # 生成報告
    report = service.generate_daily_report(results)
    
    # 保存到本地
    service.save_report_to_file(report)
    
    # 推送到配置的渠道（自動識別）
    return service.send(report)


if __name__ == "__main__":
    # 測試代碼
    logging.basicConfig(level=logging.DEBUG)
    
    # 模擬分析結果
    test_results = [
        AnalysisResult(
            code='600519',
            name='貴州茅台',
            sentiment_score=75,
            trend_prediction='看多',
            analysis_summary='技術面強勢，消息面利多',
            operation_advice='買入',
            technical_analysis='放量突破 MA20，MACD 金叉',
            news_summary='公司發布分紅公告，業績超預期',
        ),
        AnalysisResult(
            code='000001',
            name='平安銀行',
            sentiment_score=45,
            trend_prediction='震盪',
            analysis_summary='橫盤整理，等待方向',
            operation_advice='持有',
            technical_analysis='均線粘合，成交量萎縮',
            news_summary='近期無重大消息',
        ),
        AnalysisResult(
            code='300750',
            name='寧德時代',
            sentiment_score=35,
            trend_prediction='看空',
            analysis_summary='技術面走弱，注意風險',
            operation_advice='賣出',
            technical_analysis='跌破 MA10 支撐，量能不足',
            news_summary='行業競爭加劇，毛利率承壓',
        ),
    ]
    
    service = NotificationService()
    
    # 顯示檢測到的渠道
    print("=== 通知渠道檢測 ===")
    print(f"當前渠道: {service.get_channel_names()}")
    print(f"渠道列表: {service.get_available_channels()}")
    print(f"服務可用: {service.is_available()}")
    
    # 生成日報
    print("\n=== 生成日報測試 ===")
    report = service.generate_daily_report(test_results)
    print(report)
    
    # 保存到文件
    print("\n=== 保存日報 ===")
    filepath = service.save_report_to_file(report)
    print(f"保存成功: {filepath}")
    
    # 推送測試
    if service.is_available():
        print(f"\n=== 推送測試（{service.get_channel_names()}）===")
        success = service.send(report)
        print(f"推送結果: {'成功' if success else '失敗'}")
    else:
        print("\n通知渠道未配置，跳過推送測試")
