import logging
import google.generativeai as genai
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os
import json
import time
import random
import re
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Thông tin xác thực
TOKEN = '7755708665:AAEOgUu_rYrPnGFE7_BJWmr8hw9_xrZ-5e0'
GOOGLE_API_KEY = 'AIzaSyAKfwbGxOKWssWgzoOQ-U5OXAdUemEseOg'

# Cấu hình Gemini model
genai.configure(api_key=GOOGLE_API_KEY)

# Tạo nhiều cấu hình model với các nhiệt độ khác nhau
model_configs = {
    "creative": genai.GenerativeModel("gemini-2.0-flash-exp", generation_config={"temperature": 0.9, "top_p": 0.95, "top_k": 40}),
    "balanced": genai.GenerativeModel("gemini-2.0-flash-exp", generation_config={"temperature": 0.7, "top_p": 0.9, "top_k": 30}),
    "precise": genai.GenerativeModel("gemini-2.0-flash-exp", generation_config={"temperature": 0.4, "top_p": 0.8, "top_k": 20})
}

# Lưu trữ lịch sử chat và cấu hình cho mỗi người dùng
user_data = {}

# Các kỹ thuật prompt engineering tiên tiến
ADVANCED_PROMPT_TEMPLATES = [
    # Template 1: Kỹ thuật Role Play + Chain of Thought
    """System: Bạn là {role}, một chuyên gia giải quyết vấn đề có trình độ cao nhất. 
    Hãy suy nghĩ theo từng bước logic để đưa ra câu trả lời tốt nhất.
    
    Khi giải quyết vấn đề này, hãy:
    1. Phân tích kỹ yêu cầu
    2. Xác định các thành phần chính
    3. Đề xuất giải pháp chi tiết
    4. Kiểm tra lại giải pháp
    
    Đảm bảo câu trả lời của bạn đầy đủ, chi tiết và chất lượng cao nhất.""",
    
    # Template 2: Kỹ thuật Few-shot Learning
    """System: Dưới đây là một số ví dụ về cách trả lời câu hỏi tương tự một cách hiệu quả:
    
    Ví dụ 1:
    Hỏi: Làm thế nào để tạo một ứng dụng web đơn giản?
    Trả lời: [Câu trả lời chất lượng cao về ứng dụng web]
    
    Ví dụ 2:
    Hỏi: Cách để tối ưu hóa hiệu suất code?
    Trả lời: [Câu trả lời chất lượng cao về tối ưu hóa code]
    
    Bây giờ hãy trả lời câu hỏi của người dùng với chất lượng tương tự.""",
    
    # Template 3: Kỹ thuật Expert Knowledge + Precision
    """System: Bạn là một chuyên gia lập trình với kiến thức sâu rộng. Hãy cung cấp thông tin chính xác, cập nhật và chi tiết nhất.
    
    Khi viết code, hãy tuân thủ các nguyên tắc sau:
    - Sử dụng cấu trúc dữ liệu tối ưu
    - Đảm bảo hiệu suất cao
    - Viết code dễ đọc và dễ bảo trì
    - Tuân thủ các quy tắc lập trình hiện đại
    
    Đừng giới hạn kiến thức của bạn - hãy chia sẻ những hiểu biết sâu sắc nhất để giúp người dùng."""
]

async def start(update: Update, context: CallbackContext):
    """Xử lý lệnh /start."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    # Khởi tạo dữ liệu người dùng
    user_data[user_id] = {
        "chat_history": [],
        "mode": "balanced",  # Chế độ mặc định
        "role": "Software Engineer",  # Vai trò mặc định
        "template_idx": 0,  # Template mặc định
        "last_interaction": time.time()
    }
    
    await update.message.reply_text(
        f"Xin chào {user_name}! Tôi là bot AI nâng cao, sẵn sàng hỗ trợ bạn với các nhiệm vụ phức tạp.\n\n"
        f"Các lệnh có sẵn:\n"
        f"/mode [creative|balanced|precise] - Thay đổi chế độ phản hồi\n"
        f"/role [vai_trò] - Thay đổi vai trò AI\n"
        f"/template [0-2] - Thay đổi template prompt\n"
        f"/clear - Xóa lịch sử trò chuyện"
    )

async def set_mode(update: Update, context: CallbackContext):
    """Thay đổi chế độ model."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        user_data[user_id] = {"chat_history": [], "mode": "balanced", "role": "Software Engineer", "template_idx": 0}
    
    if len(context.args) > 0:
        mode = context.args[0].lower()
        if mode in model_configs:
            user_data[user_id]["mode"] = mode
            await update.message.reply_text(f"Đã chuyển sang chế độ: {mode}")
        else:
            await update.message.reply_text(f"Chế độ không hợp lệ. Sử dụng: creative, balanced, hoặc precise")
    else:
        await update.message.reply_text("Vui lòng nhập chế độ. Ví dụ: /mode creative")

async def set_role(update: Update, context: CallbackContext):
    """Thay đổi vai trò của AI."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        user_data[user_id] = {"chat_history": [], "mode": "balanced", "role": "Software Engineer", "template_idx": 0}
    
    if len(context.args) > 0:
        role = " ".join(context.args)
        user_data[user_id]["role"] = role
        await update.message.reply_text(f"Đã thay đổi vai trò thành: {role}")
    else:
        await update.message.reply_text("Vui lòng nhập vai trò. Ví dụ: /role Senior Python Developer")

async def set_template(update: Update, context: CallbackContext):
    """Thay đổi template prompt."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        user_data[user_id] = {"chat_history": [], "mode": "balanced", "role": "Software Engineer", "template_idx": 0}
    
    if len(context.args) > 0:
        try:
            template_idx = int(context.args[0])
            if 0 <= template_idx < len(ADVANCED_PROMPT_TEMPLATES):
                user_data[user_id]["template_idx"] = template_idx
                await update.message.reply_text(f"Đã thay đổi template thành: {template_idx}")
            else:
                await update.message.reply_text(f"Chỉ số template không hợp lệ. Phải từ 0-{len(ADVANCED_PROMPT_TEMPLATES)-1}")
        except ValueError:
            await update.message.reply_text("Vui lòng nhập một số nguyên")
    else:
        await update.message.reply_text(f"Vui lòng nhập chỉ số template (0-{len(ADVANCED_PROMPT_TEMPLATES)-1})")

async def clear_history(update: Update, context: CallbackContext):
    """Xóa lịch sử trò chuyện."""
    user_id = update.effective_user.id
    
    if user_id in user_data:
        user_data[user_id]["chat_history"] = []
        await update.message.reply_text("Lịch sử trò chuyện đã được xóa.")
    else:
        user_data[user_id] = {"chat_history": [], "mode": "balanced", "role": "Software Engineer", "template_idx": 0}
        await update.message.reply_text("Lịch sử trò chuyện đã được khởi tạo.")

# Hàm này sẽ được gọi thủ công thay vì tự động
async def cleanup_inactive_users(update: Update, context: CallbackContext):
    """Dọn dẹp dữ liệu người dùng không hoạt động."""
    current_time = time.time()
    inactive_threshold = 12 * 3600  # 12 giờ
    
    inactive_users = []
    for user_id, data in user_data.items():
        if "last_interaction" in data and (current_time - data["last_interaction"]) > inactive_threshold:
            inactive_users.append(user_id)
    
    for user_id in inactive_users:
        del user_data[user_id]
    
    count = len(inactive_users)
    await update.message.reply_text(f"Đã dọn dẹp dữ liệu của {count} người dùng không hoạt động")

def create_code_file(code_content, user_id, file_extension=".txt"):
    """Tạo file tạm thời chứa code."""
    file_name = f"code_{user_id}{file_extension}"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(code_content)
    return file_name

def detect_code_language(code):
    """Phát hiện ngôn ngữ lập trình để đặt đúng phần mở rộng file."""
    code_lower = code.lower()
    file_ext = ".txt"
    
    language_patterns = {
        r'\bpython\b|\bdef\s+\w+\s*\(|\bimport\s+\w+\b|\bfrom\s+\w+\s+import\b': '.py',
        r'\bjavascript\b|\bconst\b|\blet\b|\bfunction\s+\w+\s*\(|\bvar\b': '.js',
        r'\bjava\b|\bpublic\s+class\b|\bprivate\s+\w+\b|\bprotected\b': '.java',
        r'\bc\+\+\b|\bstd::\b|\bnamespace\b|\btemplate\s*<': '.cpp',
        r'\bhtml\b|<html|<body|<div': '.html',
        r'\bcss\b|\b\.\w+\s*\{': '.css',
        r'\bphp\b|\b<\?php': '.php',
        r'\bruby\b|\bdef\s+\w+\b|\brequire\s+[\'\"]': '.rb',
        r'\bgolang\b|\bgo\b|\bfunc\s+\w+\s*\(|\bpackage\s+\w+\b': '.go',
        r'\brush\b|\bfn\s+\w+\s*\(|\blet\s+mut\b': '.rs',
        r'\bswift\b|\bfunc\s+\w+\s*\(|\bvar\s+\w+\s*:\s*\w+\b': '.swift',
        r'\bkotlin\b|\bfun\s+\w+\s*\(|\bval\s+\w+\s*:\s*\w+\b': '.kt',
        r'\bsql\b|\bselect\b|\bfrom\b|\bwhere\b|\binsert\b|\bupdate\b|\bdelete\b': '.sql'
    }
    
    for pattern, extension in language_patterns.items():
        if re.search(pattern, code_lower):
            file_ext = extension
            break
    
    return file_ext

async def preprocess_input(message, user_id):
    """Tiền xử lý đầu vào để tăng tính hiệu quả."""
    # Loại bỏ các khoảng trắng không cần thiết
    message = message.strip()
    
    # Tự động thêm context nếu message quá ngắn
    if len(message) < 20 and len(user_data[user_id]["chat_history"]) > 0:
        # Thêm context từ tin nhắn cuối cùng của AI
        last_model_msg = next((msg for msg in reversed(user_data[user_id]["chat_history"]) 
                               if msg["role"] == "model"), None)
        if last_model_msg:
            message = f"Dựa trên câu trả lời cuối cùng của bạn về: '{last_model_msg['parts'][0][:100]}...', {message}"
    
    # Phát hiện và xử lý các yêu cầu code
    code_indicators = ["viết code", "làm thế nào để code", "tạo chương trình", "viết chương trình"]
    if any(indicator in message.lower() for indicator in code_indicators) and "?" in message:
        message = f"Vui lòng cung cấp code mẫu chi tiết và giải thích cho: {message}"
    
    return message

async def handle_message(update: Update, context: CallbackContext):
    """Xử lý tin nhắn đến từ người dùng."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    # Khởi tạo dữ liệu người dùng nếu chưa có
    if user_id not in user_data:
        user_data[user_id] = {
            "chat_history": [],
            "mode": "balanced",
            "role": "Software Engineer",
            "template_idx": 0,
            "last_interaction": time.time()
        }
    
    # Ghi lại thời gian tương tác
    user_data[user_id]["last_interaction"] = time.time()
    
    # Xử lý tin nhắn văn bản
    if update.message.text and not update.message.text.startswith('/'):
        message = update.message.text
        logger.info(f"Tin nhắn từ {user_name}: {message}")
        
        # Tiền xử lý đầu vào
        processed_message = await preprocess_input(message, user_id)
        
        # Thêm tin nhắn người dùng vào lịch sử
        user_data[user_id]["chat_history"].append({"role": "user", "parts": [processed_message]})
        
        try:
            # Lấy cấu hình người dùng
            mode = user_data[user_id]["mode"]
            model = model_configs[mode]
            role = user_data[user_id]["role"]
            template_idx = user_data[user_id]["template_idx"]
            
            # Áp dụng template được chọn
            template = ADVANCED_PROMPT_TEMPLATES[template_idx].format(role=role)
            
            # Chuẩn bị history trong định dạng phù hợp với Gemini
            history = [{"role": "system", "parts": [template]}]
            
            # Chỉ sử dụng 8 tin nhắn gần nhất để tiết kiệm token
            recent_history = user_data[user_id]["chat_history"][-8:]
            history.extend(recent_history)
            
            # Thêm một chút nhiễu ngẫu nhiên để tránh phản hồi lặp lại
            if random.random() < 0.2:  # 20% thời gian
                history.append({"role": "system", "parts": [f"Hãy suy nghĩ sáng tạo và đưa ra góc nhìn độc đáo về vấn đề này."]})
            
            # Gọi API Gemini
            response = model.generate_content(history)
            
            if response.text:
                # Phân tích phản hồi có chứa code
                if "```" in response.text:
                    code_blocks = response.text.split("```")[1::2]  # Lấy các đoạn code
                    
                    # Tạo và gửi file code cho mỗi đoạn
                    for i, code in enumerate(code_blocks):
                        code = code.strip()
                        
                        # Loại bỏ tên ngôn ngữ nếu có
                        if '\n' in code and not code.startswith('#'):
                            first_line = code.split('\n')[0].strip()
                            if not any(char in first_line for char in "(){}[];:,./\\"):
                                code = code.split('\n', 1)[1]
                        
                        # Phát hiện ngôn ngữ và đặt phần mở rộng file phù hợp
                        file_ext = detect_code_language(code)
                        
                        # Tạo và gửi file
                        file_name = create_code_file(code, user_id, file_ext)
                        
                        with open(file_name, "rb") as f:
                            await update.message.reply_document(
                                document=InputFile(f, filename=f"code_{i+1}{file_ext}"),
                                caption=f"Code đã tạo (phần {i+1}/{len(code_blocks)})"
                            )
                        
                        os.remove(file_name)  # Xóa file tạm
                    
                    # Gửi phần văn bản không phải code
                    remaining_text = ""
                    parts = response.text.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 0 and part.strip():
                            remaining_text += part + "\n"
                    
                    if remaining_text.strip():
                        await update.message.reply_text(remaining_text)
                else:
                    await update.message.reply_text(response.text)
                
                # Thêm phản hồi vào lịch sử
                user_data[user_id]["chat_history"].append({"role": "model", "parts": [response.text]})
                
                # Giới hạn lịch sử để tiết kiệm bộ nhớ
                if len(user_data[user_id]["chat_history"]) > 30:
                    user_data[user_id]["chat_history"] = user_data[user_id]["chat_history"][-30:]
            else:
                await update.message.reply_text("Tôi xin lỗi, tôi không thể tạo phản hồi lúc này.")
        
        except Exception as e:
            logger.error(f"Lỗi khi gọi Gemini API: {e}", exc_info=True)
            await update.message.reply_text("Đã có lỗi xảy ra. Vui lòng thử lại sau.")
    
    # Xử lý file
    elif update.message.document:
        try:
            file = await context.bot.get_file(update.message.document.file_id)
            temp_file = NamedTemporaryFile(delete=False)
            await file.download(temp_file.name)
            
            # Đọc nội dung file
            file_content = ""
            try:
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                with open(temp_file.name, 'r', encoding='latin-1') as f:
                    file_content = f.read()
            finally:
                os.remove(temp_file.name)
            
            # Giới hạn nội dung file nếu quá dài
            if len(file_content) > 15000:
                file_content = file_content[:15000] + "\n\n[Nội dung quá dài đã bị cắt ngắn...]"
            
            file_message = f"[Nội dung file {update.message.document.file_name}]\n{file_content}"
            
            # Thêm nội dung file vào lịch sử
            user_data[user_id]["chat_history"].append({"role": "user", "parts": [file_message]})
            
            # Lấy cấu hình người dùng
            mode = user_data[user_id]["mode"]
            model = model_configs[mode]
            role = user_data[user_id]["role"]
            template_idx = user_data[user_id]["template_idx"]
            
            # Áp dụng template được chọn
            template = ADVANCED_PROMPT_TEMPLATES[template_idx].format(role=role)
            
            # Chuẩn bị history
            history = [{"role": "system", "parts": [template]}]
            history.extend(user_data[user_id]["chat_history"][-8:])
            
            # Thêm chỉ dẫn đặc biệt cho xử lý file
            history.append({"role": "system", "parts": [
                f"Đây là nội dung file {update.message.document.file_name}. Hãy phân tích kỹ và đưa ra những nhận xét chi tiết."
            ]})
            
            # Gọi API Gemini
            response = model.generate_content(history)
            
            # Xử lý phản hồi tương tự như với tin nhắn văn bản
            if response.text:
                if "```" in response.text:
                    code_blocks = response.text.split("```")[1::2]
                    
                    for i, code in enumerate(code_blocks):
                        code = code.strip()
                        if '\n' in code and not code.startswith('#'):
                            first_line = code.split('\n')[0].strip()
                            if not any(char in first_line for char in "(){}[];:,./\\"):
                                code = code.split('\n', 1)[1]
                        
                        file_ext = detect_code_language(code)
                        file_name = create_code_file(code, user_id, file_ext)
                        
                        with open(file_name, "rb") as f:
                            await update.message.reply_document(
                                document=InputFile(f, filename=f"phân_tích_code_{i+1}{file_ext}"),
                                caption=f"Phân tích code (phần {i+1}/{len(code_blocks)})"
                            )
                        
                        os.remove(file_name)
                    
                    remaining_text = ""
                    parts = response.text.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 0 and part.strip():
                            remaining_text += part + "\n"
                    
                    if remaining_text.strip():
                        await update.message.reply_text(remaining_text)
                else:
                    await update.message.reply_text(response.text)
                
                user_data[user_id]["chat_history"].append({"role": "model", "parts": [response.text]})
                
                if len(user_data[user_id]["chat_history"]) > 30:
                    user_data[user_id]["chat_history"] = user_data[user_id]["chat_history"][-30:]
            else:
                await update.message.reply_text("Tôi không thể phân tích file này.")
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file: {e}", exc_info=True)
            await update.message.reply_text("Có lỗi xảy ra khi xử lý file.")

async def error(update: Update, context: CallbackContext):
    """Xử lý lỗi."""
    logger.warning(f"Update {update} gây ra lỗi {context.error}", exc_info=True)

def main():
    """Khởi tạo và chạy bot."""
    application = Application.builder().token(TOKEN).build()
    
    # Đăng ký các handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("mode", set_mode))
    application.add_handler(CommandHandler("role", set_role))
    application.add_handler(CommandHandler("template", set_template))
    application.add_handler(CommandHandler("cleanup", cleanup_inactive_users))  # Thay đổi thành lệnh thủ công
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_message))
    application.add_error_handler(error)
    
    # Không sử dụng job_queue nữa - cần cài thêm gói nếu muốn dùng
    # job_queue = application.job_queue
    # job_queue.run_repeating(cleanup_inactive_users, interval=3600)
    
    logger.info("Bot đang chạy...")
    application.run_polling()

if __name__ == '__main__':
    main()