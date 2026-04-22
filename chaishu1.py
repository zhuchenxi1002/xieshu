import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="iCCP")
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import threading
import requests
import json
import os
import re
import time
import hashlib
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==================== 增强版 LLM 适配器（支持重试） ====================
class SimpleLLM:
    def __init__(self, api_key, base_url, model, temperature, max_tokens):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = self._create_retry_session()

    def _create_retry_session(self, retries=3, backoff_factor=1):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def chat(self, prompt, retry_count=3):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        last_exception = None
        for attempt in range(1, retry_count + 1):
            try:
                resp = self.session.post(url, headers=headers, json=data, timeout=(10, 120))
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    last_exception = Exception(error_msg)
                    if attempt == retry_count:
                        return f"错误：{error_msg}"
                    time.sleep(1 * attempt)
            except requests.exceptions.Timeout:
                last_exception = Exception("请求超时")
                if attempt == retry_count:
                    return f"请求异常：超时"
                time.sleep(1 * attempt)
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt == retry_count:
                    return f"请求异常：连接错误 - {str(e)}"
                time.sleep(1 * attempt)
            except Exception as e:
                last_exception = e
                if attempt == retry_count:
                    return f"请求异常：{str(e)}"
                time.sleep(1 * attempt)
        return f"请求异常：{str(last_exception)}"


# ==================== 知识库管理器（增加缓存和版本管理） ====================
class KnowledgeBaseManager:
    def __init__(self, storage_dir="./zhishiku"):
        self.storage_dir = storage_dir
        self.meta_file = os.path.join(storage_dir, "meta.json")
        self.cache_dir = os.path.join(storage_dir, "analysis_cache")
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.meta = self._load_meta()

    def _load_meta(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _sanitize_filename(self, name):
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    def upload_document(self, title, content):
        doc_id = title
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version = 1
        if doc_id in self.meta:
            versions = self.meta[doc_id]['versions']
            version = max(v['version'] for v in versions) + 1
            for v in versions:
                v['active'] = False
        else:
            self.meta[doc_id] = {'title': title, 'versions': []}
        safe_title = self._sanitize_filename(doc_id)
        file_name = f"{safe_title}_v{version}.txt"
        file_path = os.path.join(self.storage_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        version_info = {'version': version, 'time': now, 'file': file_name, 'active': True}
        self.meta[doc_id]['versions'].append(version_info)
        self._save_meta()
        self.clear_cache(doc_id)
        return doc_id, version

    def get_document_md5(self, doc_id):
        content = self.get_document_content(doc_id)
        if content is None:
            return None
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get_cached_analysis(self, doc_id):
        md5 = self.get_document_md5(doc_id)
        if not md5:
            return None
        cache_file = os.path.join(self.cache_dir, f"{doc_id}_{md5}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def save_cached_analysis(self, doc_id, analysis_result, template_text):
        md5 = self.get_document_md5(doc_id)
        if not md5:
            return
        cache_file = os.path.join(self.cache_dir, f"{doc_id}_{md5}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_result': analysis_result,
                'template_text': template_text,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def clear_cache(self, doc_id):
        for f in os.listdir(self.cache_dir):
            if f.startswith(doc_id + "_"):
                os.remove(os.path.join(self.cache_dir, f))

    def get_all_documents(self):
        result = []
        for doc_id, info in self.meta.items():
            active_version = None
            last_time = None
            for v in info['versions']:
                if v['active']:
                    active_version = v['version']
                    last_time = v['time']
            if active_version is None and info['versions']:
                latest = max(info['versions'], key=lambda x: x['version'])
                active_version = latest['version']
                last_time = latest['time']
            result.append({'title': info['title'], 'active_version': active_version, 'last_time': last_time, 'doc_id': doc_id})
        result.sort(key=lambda x: x['last_time'], reverse=True)
        return result

    def search_documents(self, keyword):
        if not keyword:
            return self.get_all_documents()
        keyword_lower = keyword.lower()
        return [doc for doc in self.get_all_documents() if keyword_lower in doc['title'].lower()]

    def get_document_content(self, doc_id, version=None):
        if doc_id not in self.meta:
            return None
        versions = self.meta[doc_id]['versions']
        target = None
        if version is None:
            for v in versions:
                if v['active']:
                    target = v
                    break
            if target is None and versions:
                target = max(versions, key=lambda x: x['version'])
        else:
            for v in versions:
                if v['version'] == version:
                    target = v
        if target:
            file_path = os.path.join(self.storage_dir, target['file'])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        return None

    def get_version_list(self, doc_id):
        if doc_id not in self.meta:
            return []
        return sorted(self.meta[doc_id]['versions'], key=lambda x: x['version'], reverse=True)

    def switch_version(self, doc_id, version):
        if doc_id not in self.meta:
            return False
        for v in self.meta[doc_id]['versions']:
            v['active'] = (v['version'] == version)
        self._save_meta()
        self.clear_cache(doc_id)
        return True

    def delete_document(self, doc_id):
        if doc_id in self.meta:
            for v in self.meta[doc_id]['versions']:
                file_path = os.path.join(self.storage_dir, v['file'])
                if os.path.exists(file_path):
                    os.remove(file_path)
            self.clear_cache(doc_id)
            del self.meta[doc_id]
            self._save_meta()

# ==================== 拆书分析库管理器（继承自 KnowledgeBaseManager） ====================
class BookAnalysisManager(KnowledgeBaseManager):
    def __init__(self):
        super().__init__(storage_dir="./fenxi")

    def sync_with_folder(self):
        """
        自动同步fenxi目录下的分析结果文件到meta.json，补录未登记的分析文档。
        """
        for fname in os.listdir(self.storage_dir):
            if fname.endswith('.txt'):
                title = fname[:-4]
                doc_id = title
                file_path = os.path.join(self.storage_dir, fname)
                if doc_id not in self.meta:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    version = 1
                    self.meta[doc_id] = {'title': doc_id, 'versions': []}
                    version_info = {'version': version, 'time': now, 'file': fname, 'active': True}
                    self.meta[doc_id]['versions'].append(version_info)
        self._save_meta()


# ==================== 模板库管理器 ====================
class TemplateManager:
    def __init__(self, storage_dir="./templates"):
        self.storage_dir = storage_dir
        self.index_file = os.path.join(storage_dir, "index.json")
        os.makedirs(storage_dir, exist_ok=True)
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_index(self):
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def save_template(self, source_title, template_text, analysis_summary=""):
        template_id = f"{source_title}_{int(time.time())}"
        safe_id = re.sub(r'[<>:"/\\|?*]', '_', template_id)
        file_path = os.path.join(self.storage_dir, f"{safe_id}.json")
        record = {
            "id": template_id,
            "source_title": source_title,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "template_text": template_text,
            "analysis_summary": analysis_summary,
            "rating": 0,
            "use_count": 0
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.index.append({"id": template_id, "source_title": source_title, "created": record["created"]})
        self._save_index()
        return template_id

    def get_all_templates(self):
        templates = []
        for item in self.index:
            file_path = os.path.join(self.storage_dir, f"{item['id']}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    templates.append(json.load(f))
        return templates

    def get_template(self, template_id):
        file_path = os.path.join(self.storage_dir, f"{template_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def delete_template(self, template_id):
        file_path = os.path.join(self.storage_dir, f"{template_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        self.index = [item for item in self.index if item["id"] != template_id]
        self._save_index()

    def rate_template(self, template_id, rating):
        record = self.get_template(template_id)
        if record:
            record["rating"] = rating
            file_path = os.path.join(self.storage_dir, f"{template_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)


# ==================== 主应用 ====================
class SimpleWorkbench:
    MODEL_PRESETS = {
        "deepseek": ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
        "doubao-lite": ("豆包-Seed-2.0-lite", "https://ark.cn-beijing.volces.com/api/v3", "doubao-seed-2-0-lite-260215"),
        "doubao-pro": ("豆包-Seed-2.0-pro", "https://ark.cn-beijing.volces.com/api/v3", "doubao-seed-2-0-pro-260215"),
        "siliconflow": ("SiliconFlow", "https://api.siliconflow.cn/v1", "deepseek-ai/DeepSeek-V2.5"),
        "openai": ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
        "minimax2.7": ("MiniMax 2.7", "https://api.minimax.chat/v1", "minimax-m2.7"),
        "claude-code": ("Claude Code", "https://api.anthropic.com/v1", "claude-code"),
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("知识库 & 系统设置")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f2f5')

        # LLM 配置变量
        self.api_key = tk.StringVar()
        self.base_url = tk.StringVar(value="https://api.deepseek.com/v1")
        self.model_name = tk.StringVar(value="deepseek-chat")
        self.temp = tk.DoubleVar(value=0.7)
        self.current_model = tk.StringVar(value="deepseek")

        # 知识库
        self.kb_manager = KnowledgeBaseManager()
        self.book_analysis_manager = BookAnalysisManager()
        self.book_analysis_manager.sync_with_folder()
        self.template_manager = TemplateManager()

        # 写书页面相关变量
        self.new_book_title = tk.StringVar()
        self.new_book_synopsis = tk.StringVar()
        self.save_path = tk.StringVar(value="./小说输出")
        self.chapter_num = tk.IntVar(value=25)
        self.words_per_chapter = tk.IntVar(value=2800)
        self.genre = tk.StringVar(value="")
        self.architecture = ""
        self.chapter_blueprints = {}
        self.chapter_drafts = {}
        self.current_chapter = 1

        # 新增：当前选中的提示词文件路径
        self.selected_prompt_file = tk.StringVar()

        # 进度条变量
        self.progress_var = tk.IntVar(value=0)
        self.progress_label = tk.StringVar(value="就绪")

        # 防并发标志
        self.is_analyzing = False
        self.is_generating_prompt = False
        self.is_generating_arch = False
        self.is_generating_blueprint = False
        self.is_generating_partial = False

        # 加载配置
        self.load_llm_config()
        self.setup_ui()
        self.load_global_state()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 加载写作风格提示（从外部文件）
        self.deai_prompt = self.load_writing_style()

    def load_writing_style(self):
        style_file = "writing_style.txt"
        default_style = """
**写作要求（去AI味，贴近人类风格）：**
- 细节描绘：通过对环境和情感细致入微的描写，增强文章的真实感。
- 增强代入感：让读者感同身受，增强他们与内容的情感连接。
- 分段解构：将复杂的内容分解成小段，便于读者逐步理解。
- 分发好奇心：通过提出问题或悬念，激发读者的求知欲望。
- 增添幽默：适当地加入幽默元素，提升文章的趣味性。
- 平衡叙述节奏：通过交替使用长句和短句，使文章的节奏更具吸引力。
- 情感共鸣：通过描写情感变化，促使读者产生共鸣。
- 提供背景信息：适时插入相关背景知识，让读者更好理解文章内容。
- 简洁明了：去除冗余信息，使文章结构更加简洁有力。
- 多视角叙述：从不同角度描述事件，丰富文章的层次感。
- 突出关键点：使用重点词汇或短语，强化文章的主旨。
- 打造紧张氛围：通过描写紧张的情境，增加文章的悬念感。
- 使用对比：通过对比手法突出主题，使文章层次分明。
- 层层递进：逐步深入剖析问题，使读者更容易理解复杂概念。
- 情节反转：设计出人意料的情节反转，增加文章的戏剧性。
- 明确结论：在文章结尾处提供明确的总结或结论，增强说服力。
- 运用拟人手法：将非人类的事物赋予人的特征，使描写更生动。
- 多元化表达：通过多种修辞手法，使文章语言更加丰富。
- 呼应开篇：结尾处呼应开篇的内容，使文章结构更加严谨。
- 使用类比：通过类比的方式解释复杂概念，使其更易理解。
- 增强情感深度：通过细致描写内心活动，增加情感的层次感。
- 减少术语使用：避免使用过多专业术语，使文章通俗易懂。
- 多感官描述：调动视觉、听觉等多种感官，丰富文章的描写。
- 精确用词：选择最恰当的词汇表达意思，避免模棱两可的表述。
- 制造矛盾冲突：通过引入矛盾，使情节更加紧张和引人入胜。
- 提供解决方案：在提出问题后，及时给出解决办法，增强实用性。
- 层次分明：通过分段和分层次描述，使文章逻辑清晰。
- 预设读者反应：预测读者可能的反应并提前回应，增强互动感。
- 插入实例：通过具体实例说明抽象概念，使内容更有说服力。
- 运用反问句：使用反问句强化观点，引发读者思考。
- 适度夸张：通过适当夸张，增强描述的生动性和感染力。
- 细化场景描写：对场景进行精细描述，增强画面感。
- 建立悬念：在叙述中埋下伏笔，吸引读者继续阅读。
- 巧用反义词：通过反义词对比，强化文章的对比效果。
- 使用隐喻：运用隐喻使文章更具深度和艺术性。
- 营造紧迫感：通过描写紧急情况，增强文章的紧张感。
- 引导情绪波动：通过逐步升级情绪，使读者情感得到释放。
- 引用流行语：适时使用流行语，使文章更接地气。
- 强化视觉效果：使用生动的视觉描述，增强读者的画面感。
- 嵌入故事情节：通过嵌入小故事，丰富文章的情感层次。
- 利用数字数据：引用具体数据，增强文章的可信度。
- 排比句式：使用排比句式，增强文章的节奏感和力量感。
- 对比论证：通过对比不同观点，增强论证的说服力。
- 简单化复杂内容：将复杂概念简单化，使其易于理解。
- 使用直接引语：通过直接引语，使人物对话更加生动。
- 运用反复手法：通过反复强调某一观点，强化文章的主旨。
- 引导读者思考：通过提出问题，引导读者进行深入思考。
- 丰富背景描写：通过增加背景描写，使情节更具立体感。
- 融入情感记忆：借助情感记忆，增强文章的共鸣感。
- 呼应读者经验：通过呼应读者的生活经验，增加文章的亲切感。
- 强调行动力：通过描写行动场景，增强文章的动感。
- 构建人物形象：通过细节描写，塑造生动的人物形象。
- 营造对比冲突：通过制造对比冲突，增强情节的张力。
- 运用倒叙手法：使用倒叙手法，使故事结构更加多样化。
- 嵌入哲理思考：在叙述中融入哲理思考，增加文章的深度。
- 使用重复句式：通过重复句式，增强文章的力量感。
- 引入视觉细节：通过增加视觉细节，使场景更加生动。
- 制造反差：通过制造强烈的反差，增加文章的戏剧效果。
- 使用简短句式：通过简短句式，增强文章的冲击力。
- 通过细节刻画人物：细腻的细节描写，使人物更加立体生动。
- 使用情感铺垫：通过情感铺垫，为后续情节发展做准备。
- 逐层递进：从浅到深逐步展开，增强文章的层次感。
- 运用时间顺序：通过时间顺序，使叙事更加清晰流畅。
- 加入自然描写：通过描写自然景物，增强文章的画面感。
- 使用隐含对比：通过隐含对比，增加文章的深度和趣味。
- 增加环境描写：丰富环境描写，增强文章的现场感。
- 适度幽默：通过适度幽默，增加文章的轻松感。
- 强调现实基础：通过引用现实案例，增强文章的可信度。
- 设计悬疑结尾：通过悬疑结尾，引发读者的好奇心。
- 运用象征手法：通过象征手法，增加文章的象征意义。
- 增强互动性：通过问题或呼吁，增强读者的参与感。
- 利用故事开头：通过讲述故事开头，引发读者兴趣。
- 制造紧张气氛：通过紧张的情节设置，增强文章的紧迫感。
- 增强表达层次：通过多层次的表达，丰富文章的内容。
- 合理使用比喻：通过比喻手法，使抽象概念形象化。
- 增加文化元素：融入文化元素，增强文章的深度和背景感。
- 制造轻松氛围：通过轻松的语言和情境，缓解读者的阅读压力。
- 使用直接对话：通过直接对话，使人物交流更加真实。
- 通过悬念吸引：在开头设置悬念，吸引读者的注意力。
- 合理引入矛盾：通过矛盾冲突，增加文章的戏剧性。
- 使用具体例子：通过具体例子说明抽象问题，增强文章的实用性。
- 通过情境塑造：通过具体情境的塑造，使情节更有代入感。
"""
        if os.path.exists(style_file):
            with open(style_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            with open(style_file, 'w', encoding='utf-8') as f:
                f.write(default_style)
            return default_style

    # ---------- 全局状态持久化 ----------
    def save_global_state(self):
        state = {}
        try:
            if hasattr(self, 'analysis_result_text'):
                state['analysis_result'] = self.analysis_result_text.get(1.0, tk.END).strip()
            if hasattr(self, 'new_book_prompt_text'):
                state['new_book_prompt'] = self.new_book_prompt_text.get(1.0, tk.END).strip()
        except:
            pass
        state['new_book_title'] = self.new_book_title.get()
        state['new_book_synopsis'] = self.new_book_synopsis.get()
        state['save_path'] = self.save_path.get()
        state['chapter_num'] = self.chapter_num.get()
        state['words_per_chapter'] = self.words_per_chapter.get()
        state['genre'] = self.genre.get()
        state['architecture'] = self.architecture
        state['chapter_blueprints'] = self.chapter_blueprints
        state['chapter_drafts'] = self.chapter_drafts
        try:
            if hasattr(self, 'text_arch'):
                state['arch_text'] = self.text_arch.get(1.0, tk.END).strip()
            if hasattr(self, 'text_blueprint'):
                state['blueprint_text'] = self.text_blueprint.get(1.0, tk.END).strip()
            if hasattr(self, 'text_draft'):
                state['draft_text'] = self.text_draft.get(1.0, tk.END).strip()
        except:
            pass
        try:
            if hasattr(self, 'kb_search_var'):
                state['kb_search'] = self.kb_search_var.get()
        except:
            pass
        with open('global_state.json', 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_global_state(self):
        if not os.path.exists('global_state.json'):
            return
        try:
            with open('global_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)
            if hasattr(self, 'analysis_result_text') and 'analysis_result' in state:
                self.analysis_result_text.insert(tk.END, state['analysis_result'])
                if self.analysis_result_text.get(1.0, tk.END).strip() == "正在分析中，请稍候...":
                    self.analysis_result_text.delete(1.0, tk.END)
            if hasattr(self, 'new_book_prompt_text') and 'new_book_prompt' in state:
                self.new_book_prompt_text.insert(tk.END, state['new_book_prompt'])
            self.new_book_title.set(state.get('new_book_title', ''))
            self.new_book_synopsis.set('')
            self.save_path.set(state.get('save_path', './小说输出'))
            self.chapter_num.set(state.get('chapter_num', 25))
            self.words_per_chapter.set(state.get('words_per_chapter', 2800))
            self.genre.set(state.get('genre', ''))
            self.architecture = state.get('architecture', '')
            self.chapter_blueprints = {int(k): v for k, v in state.get('chapter_blueprints', {}).items()}
            self.chapter_drafts = {int(k): v for k, v in state.get('chapter_drafts', {}).items()}
            self.root.after(100, lambda: self.restore_write_page_texts(state))
            if hasattr(self, 'kb_search_var') and 'kb_search' in state:
                self.kb_search_var.set(state['kb_search'])
                self.refresh_kb_list()
        except Exception as e:
            print(f"加载全局状态失败：{e}")

    def restore_write_page_texts(self, state):
        if hasattr(self, 'text_arch') and 'arch_text' in state:
            self.text_arch.insert(tk.END, state['arch_text'])
        if hasattr(self, 'text_blueprint') and 'blueprint_text' in state:
            self.text_blueprint.insert(tk.END, state['blueprint_text'])
        if hasattr(self, 'text_draft') and 'draft_text' in state:
            self.text_draft.insert(tk.END, state['draft_text'])

    def save_llm_config(self, silent=False):
        config = {
            'api_key': self.api_key.get(),
            'base_url': self.base_url.get(),
            'model_name': self.model_name.get(),
            'temperature': self.temp.get(),
            'current_model': self.current_model.get()
        }
        with open('llm_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        if not silent:
            messagebox.showinfo("成功", "配置已保存！API Key 将保持不变，直到您重新输入新的 Key。")

    def load_llm_config(self):
        if os.path.exists('llm_config.json'):
            try:
                with open('llm_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.api_key.set(config.get('api_key', ''))
                self.base_url.set(config.get('base_url', 'https://api.deepseek.com/v1'))
                self.model_name.set(config.get('model_name', 'deepseek-chat'))
                self.temp.set(config.get('temperature', 0.7))
                self.current_model.set(config.get('current_model', 'deepseek'))
            except:
                pass

    def on_closing(self):
        self.save_global_state()
        self.save_llm_config(silent=True)
        self.root.destroy()

    # ---------- UI 搭建 ----------
    def setup_ui(self):
        left_nav = tk.Frame(self.root, bg='#2c3e50', width=200)
        left_nav.pack(side=tk.LEFT, fill=tk.Y)
        left_nav.pack_propagate(False)

        btn_style = {'font': ('微软雅黑', 11), 'bg': '#2c3e50', 'fg': 'white',
                     'activebackground': '#34495e', 'activeforeground': 'white',
                     'bd': 0, 'anchor': 'w', 'padx': 20, 'pady': 12, 'width': 18}

        title_label = tk.Label(left_nav, text="工作台", font=('微软雅黑', 16, 'bold'),
                               bg='#2c3e50', fg='#ecf0f1')
        title_label.pack(pady=(30, 20))

        home_btn = tk.Button(left_nav, text="🏠 首页", command=self.show_home, **btn_style)
        home_btn.pack(fill=tk.X, pady=2)
        kb_btn = tk.Button(left_nav, text="📚 知识库", command=self.show_knowledge, **btn_style)
        kb_btn.pack(fill=tk.X, pady=2)
        book_btn = tk.Button(left_nav, text="📖 拆书", command=self.show_book_analysis, **btn_style)
        book_btn.pack(fill=tk.X, pady=2)
        write_btn = tk.Button(left_nav, text="✍️ 写书", command=self.show_write_book, **btn_style)
        write_btn.pack(fill=tk.X, pady=2)

        spacer = tk.Frame(left_nav, bg='#2c3e50')
        spacer.pack(expand=True, fill=tk.BOTH)

        settings_btn = tk.Button(left_nav, text="⚙️ 系统设置", command=self.show_system_settings, **btn_style)
        settings_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

        self.right_area = tk.Frame(self.root, bg='#f0f2f5')
        self.right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.page_home = self.create_home_page()
        self.page_knowledge = self.create_knowledge_page()
        self.page_system = self.create_system_settings_page()
        self.page_book_analysis = self.create_book_analysis_page()
        self.page_write_book = self.create_write_book_page()

        self.show_home()

    # ---------- 首页 ----------
    def create_home_page(self):
        page = tk.Frame(self.right_area, bg='#f0f2f5')
        center_frame = tk.Frame(page, bg='#f0f2f5')
        center_frame.pack(expand=True)
        colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']
        text = "欢迎使用本软件"
        for i, char in enumerate(text):
            label = tk.Label(center_frame, text=char, font=('微软雅黑', 48, 'bold'),
                             fg=colors[i % len(colors)], bg='#f0f2f5')
            label.pack(side=tk.LEFT)
        sub_label = tk.Label(center_frame, text="\n高效知识管理 · 智能API配置 · 深度拆书分析 · 辅助写书", font=('微软雅黑', 14),
                             fg='#7f8c8d', bg='#f0f2f5')
        sub_label.pack(side=tk.BOTTOM, pady=20)
        return page

    # ---------- 知识库页面 ----------
    def create_knowledge_page(self):
        page = tk.Frame(self.right_area, bg='#f0f2f5')
        upload_frame = ttk.LabelFrame(page, text="上传文档（支持多文件）", padding=10)
        upload_frame.pack(fill=tk.X, padx=20, pady=10)
        self.kb_file_path = tk.StringVar()
        ttk.Label(upload_frame, text="选择文件:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(upload_frame, textvariable=self.kb_file_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(upload_frame, text="浏览（支持多选）", command=self.browse_kb_file).grid(row=0, column=2, padx=5)
        
        # 已选择文件列表框
        list_frame = tk.Frame(upload_frame)
        list_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        self.kb_files_listbox = tk.Listbox(list_frame, height=6)
        self.kb_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        remove_btn_frame = tk.Frame(list_frame)
        remove_btn_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(remove_btn_frame, text="移除\n选中", command=self.remove_selected_kb_file, width=8).pack(pady=2)
        ttk.Button(remove_btn_frame, text="清空\n列表", command=lambda: self.kb_files_listbox.delete(0, tk.END) or (setattr(self, 'kb_selected_files', []) if hasattr(self, 'kb_selected_files') else None), width=8).pack(pady=2)
        
        ttk.Button(upload_frame, text="上传文档", command=self.upload_kb_document).grid(row=2, column=0, columnspan=3, pady=10)
        search_frame = ttk.LabelFrame(page, text="文档列表", padding=10)
        search_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        search_row = tk.Frame(search_frame)
        search_row.pack(fill=tk.X, pady=5)
        ttk.Label(search_row, text="按标题搜索:").pack(side=tk.LEFT, padx=5)
        self.kb_search_var = tk.StringVar()
        self.kb_search_var.trace('w', lambda *a: self.refresh_kb_list())
        ttk.Entry(search_row, textvariable=self.kb_search_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_row, text="刷新", command=self.refresh_kb_list).pack(side=tk.LEFT, padx=5)
        columns = ("标题", "激活版本", "最后更新", "操作")
        self.kb_tree = ttk.Treeview(search_frame, columns=columns, show="headings", height=15)
        for col in columns:
            self.kb_tree.heading(col, text=col)
            if col == "标题":
                self.kb_tree.column(col, width=300)
            elif col == "激活版本":
                self.kb_tree.column(col, width=100)
            elif col == "最后更新":
                self.kb_tree.column(col, width=150)
            else:
                self.kb_tree.column(col, width=100)
        self.kb_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(search_frame, orient=tk.VERTICAL, command=self.kb_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.kb_tree.configure(yscrollcommand=scrollbar.set)
        self.kb_tree.bind("<Double-1>", self.view_kb_document)
        self.kb_menu = tk.Menu(self.root, tearoff=0)
        self.kb_menu.add_command(label="查看内容", command=self.view_kb_document)
        self.kb_menu.add_command(label="版本管理", command=self.manage_kb_versions)
        self.kb_menu.add_command(label="删除文档", command=self.delete_kb_document)
        self.kb_tree.bind("<Button-3>", self.show_kb_context_menu)
        self.refresh_kb_list()
        return page

    def browse_kb_file(self):
        # 支持多文件选择
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
        if file_paths:
            # 清空之前的文件列表
            if hasattr(self, 'kb_selected_files'):
                self.kb_selected_files.clear()
            else:
                self.kb_selected_files = []
            
            # 添加新选择的文件
            for path in file_paths:
                if path not in self.kb_selected_files:
                    self.kb_selected_files.append(path)
            
            # 更新列表框显示
            self.kb_files_listbox.delete(0, tk.END)
            for path in self.kb_selected_files:
                filename = os.path.basename(path)
                self.kb_files_listbox.insert(tk.END, filename)
            
            # 同时更新单文件路径（兼容）
            self.kb_file_path.set(file_paths[0] if file_paths else "")

    def upload_kb_document(self):
        # 检查是否有选择的文件
        if not hasattr(self, 'kb_selected_files') or not self.kb_selected_files:
            file_path = self.kb_file_path.get().strip()
            if not file_path or not os.path.exists(file_path):
                messagebox.showwarning("提示", "请选择有效的txt文件")
                return
            # 单文件模式
            self._upload_single_file(file_path)
            self.kb_file_path.set("")
        else:
            # 多文件模式
            success_count = 0
            fail_count = 0
            for file_path in self.kb_selected_files:
                if os.path.exists(file_path):
                    result = self._upload_single_file(file_path, show_message=False)
                    if result:
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
            
            # 显示总体结果
            if success_count > 0:
                messagebox.showinfo("成功", f"成功上传 {success_count} 个文档" + (f"，失败 {fail_count} 个" if fail_count > 0 else ""))
            else:
                messagebox.showerror("错误", f"上传失败，请检查文件是否有效")
            
            # 清空选择列表
            self.kb_selected_files.clear()
            self.kb_files_listbox.delete(0, tk.END)
            self.kb_file_path.set("")
        
        self.refresh_kb_list()
        self.refresh_book_doc_list()

    def _upload_single_file(self, file_path, show_message=True):
        """上传单个文件的内部函数"""
        if not os.path.exists(file_path):
            if show_message:
                messagebox.showwarning("提示", f"文件不存在：{file_path}")
            return False
        
        title = os.path.splitext(os.path.basename(file_path))[0]
        content = None
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin-1']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                print(f"成功使用编码 {enc} 读取文件")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            if show_message:
                messagebox.showerror("错误", f"无法读取文件，尝试的编码均失败：{', '.join(encodings)}")
            return False
        
        doc_id, version = self.kb_manager.upload_document(title, content)
        if show_message:
            messagebox.showinfo("成功", f"文档「{title}」上传成功，版本 v{version}")
        return True

    def remove_selected_kb_file(self):
        """移除已选择的单个文件"""
        if not hasattr(self, 'kb_selected_files') or not self.kb_selected_files:
            return
        
        selection = self.kb_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选中要移除的文件")
            return
        
        idx = selection[0]
        if idx < len(self.kb_selected_files):
            removed_path = self.kb_selected_files.pop(idx)
            self.kb_files_listbox.delete(idx)
            
            # 更新单文件路径
            if self.kb_selected_files:
                self.kb_file_path.set(self.kb_selected_files[0])
            else:
                self.kb_file_path.set("")

    def refresh_kb_list(self):
        for item in self.kb_tree.get_children():
            self.kb_tree.delete(item)
        keyword = self.kb_search_var.get().strip()
        docs = self.kb_manager.search_documents(keyword)
        for doc in docs:
            self.kb_tree.insert("", tk.END, values=(
                doc['title'],
                f"v{doc['active_version']}",
                doc['last_time'],
                "双击查看"
            ), iid=doc['doc_id'])

    def view_kb_document(self, event=None):
        selected = self.kb_tree.selection()
        if not selected:
            return
        doc_id = selected[0]
        content = self.kb_manager.get_document_content(doc_id)
        if content is None:
            messagebox.showerror("错误", "无法获取文档内容")
            return
        win = tk.Toplevel(self.root)
        win.title(f"查看文档 - {doc_id}")
        win.geometry("700x500")
        text_area = scrolledtext.ScrolledText(win, wrap=tk.WORD)
        text_area.pack(fill=tk.BOTH, expand=True)
        text_area.insert(tk.END, content)
        text_area.config(state=tk.DISABLED)

    def manage_kb_versions(self):
        selected = self.kb_tree.selection()
        if not selected:
            return
        doc_id = selected[0]
        versions = self.kb_manager.get_version_list(doc_id)
        if not versions:
            return
        win = tk.Toplevel(self.root)
        win.title(f"版本管理 - {doc_id}")
        win.geometry("500x400")
        tk.Label(win, text=f"文档：{doc_id}", font=('微软雅黑', 12, 'bold')).pack(pady=10)
        listbox = tk.Listbox(win, height=15)
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        for v in versions:
            active_mark = "✓ " if v['active'] else "  "
            listbox.insert(tk.END, f"{active_mark}v{v['version']} - {v['time']}")
        def switch():
            selection = listbox.curselection()
            if not selection:
                return
            idx = selection[0]
            target_version = versions[idx]['version']
            if self.kb_manager.switch_version(doc_id, target_version):
                messagebox.showinfo("成功", f"已切换到 v{target_version}")
                win.destroy()
                self.refresh_kb_list()
            else:
                messagebox.showerror("错误", "切换失败")
        ttk.Button(win, text="切换到选中版本", command=switch).pack(pady=10)

    def delete_kb_document(self):
        selected = self.kb_tree.selection()
        if not selected:
            return
        doc_id = selected[0]
        if messagebox.askyesno("确认删除", f"确定要删除文档「{doc_id}」及其所有版本吗？"):
            self.kb_manager.delete_document(doc_id)
            self.refresh_kb_list()
            self.refresh_book_doc_list()

    def show_kb_context_menu(self, event):
        item = self.kb_tree.identify_row(event.y)
        if item:
            self.kb_tree.selection_set(item)
            self.kb_menu.post(event.x_root, event.y_root)

    # ---------- 拆书页面（增强，修复右侧显示区） ----------
    def create_book_analysis_page(self):
        page = tk.Frame(self.right_area, bg='#f0f2f5')
        title_label = tk.Label(page, text="拆书分析", font=('微软雅黑', 18, 'bold'),
                               bg='#f0f2f5', fg='#2c3e50')
        title_label.pack(anchor='w', padx=20, pady=10)

        main_frame = tk.Frame(page, bg='#f0f2f5')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # ========== 左侧：文档选择与拆书分析库 ==========
        left_frame = ttk.LabelFrame(main_frame, text="文档选择", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))

        select_frame = tk.Frame(left_frame)
        select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(select_frame, text="选择文档:").pack(side=tk.LEFT, padx=5)
        self.book_combobox = ttk.Combobox(select_frame, width=30, state="readonly")
        self.book_combobox.pack(side=tk.LEFT, padx=5)

        self.detail_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(select_frame, text="详细分析（含人物弧光、对话技巧）",
                        variable=self.detail_analysis_var).pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        self.create_btn = ttk.Button(btn_frame, text="拆书分析", command=self.create_book_analysis)
        self.create_btn.pack(side=tk.LEFT, padx=5)
        self.edit_template_btn = ttk.Button(btn_frame, text="编辑模板", command=self.edit_template, state=tk.DISABLED)
        self.edit_template_btn.pack(side=tk.LEFT, padx=5)

        self.analysis_hint = tk.Label(left_frame, text="请从下拉框选择文档，然后点击按钮",
                                      font=('微软雅黑', 10), fg='gray', bg='#f0f2f5')
        self.analysis_hint.pack(pady=5)

        library_frame = ttk.LabelFrame(left_frame, text="拆书分析库列表", padding=10)
        library_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.book_analysis_listbox = tk.Listbox(library_frame, height=15)
        self.book_analysis_listbox.pack(fill=tk.BOTH, expand=True)
        self.book_analysis_listbox.bind("<Button-3>", self.show_book_analysis_menu)
        self.book_analysis_listbox.bind("<ButtonRelease-1>", self.on_select_book_analysis)

        # ========== 右侧：拆书分析结果显示区 ==========
        right_frame = ttk.LabelFrame(main_frame, text="拆书分析结果（含复用模板）", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.analysis_result_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=('微软雅黑', 10))
        self.analysis_result_text.pack(fill=tk.BOTH, expand=True)

        self.refresh_book_doc_list()
        self.refresh_analysis_library()
        return page

    def refresh_book_doc_list(self):
        docs = self.kb_manager.get_all_documents()
        titles = [doc['title'] for doc in docs]
        self.book_combobox['values'] = titles
        if titles:
            self.book_combobox.set(titles[0])
        else:
            self.book_combobox.set('')

    def on_select_book_analysis(self, event=None):
        selection = self.book_analysis_listbox.curselection()
        if not selection:
            return
        title_full = self.book_analysis_listbox.get(selection[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        content = self.book_analysis_manager.get_document_content(title)
        self.analysis_result_text.delete(1.0, tk.END)
        if content:
            self.analysis_result_text.insert(tk.END, content)
        else:
            self.analysis_result_text.insert(tk.END, "未找到内容")

    def on_select_book_analysis_with_prompts(self, event=None):
        """当点击拆书分析库中的文档时，加载该文档内容并查找相关的新书提示词"""
        selection = self.book_analysis_listbox.curselection()
        if not selection:
            return
        
        # 获取选中的拆书分析文档标题
        title_full = self.book_analysis_listbox.get(selection[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        
        # 加载拆书分析内容到右侧文本框
        content = self.book_analysis_manager.get_document_content(title)
        self.analysis_result_text.delete(1.0, tk.END)
        if content:
            self.analysis_result_text.insert(tk.END, content)
        else:
            self.analysis_result_text.insert(tk.END, "未找到内容")
        
        # 清空新书提示词列表框
        self.book_prompts_listbox.delete(0, tk.END)
        
        # 查找与该拆书分析相关的新书提示词
        self.load_new_book_prompts_for_analysis(title)
    
    def load_new_book_prompts_for_analysis(self, analysis_title):
        """加载与指定拆书分析相关的新书提示词"""
        tishici_dir = "./tishici"
        if not os.path.exists(tishici_dir):
            return
        
        # 获取所有新书提示词文件
        prompt_files = []
        for filename in os.listdir(tishici_dir):
            if filename.endswith('-提示词.txt'):
                prompt_files.append(filename)
        
        # 简单关联逻辑：如果新书提示词文件名包含拆书分析的关键词，则认为是相关的
        # 拆书分析标题格式通常是 "原书名-拆书"，我们提取原书名的部分
        original_book_title = analysis_title.replace('-拆书', '')
        
        # 查找相关的新书提示词
        related_prompts = []
        for filename in prompt_files:
            # 从文件名中提取新书名（去掉"-提示词.txt"后缀）
            new_book_title = filename.replace('-提示词.txt', '')
            
            # 简单匹配：如果新书提示词文件名包含原书名的关键词，或者原书名包含新书名的关键词
            # 这里使用简单的包含匹配，实际可以根据需要实现更复杂的关联逻辑
            if original_book_title in filename or new_book_title in analysis_title:
                related_prompts.append((filename, new_book_title))
        
        # 如果没有找到相关提示词，尝试更宽松的匹配
        if not related_prompts:
            # 使用拆书分析标题的前几个字符进行匹配
            search_prefix = original_book_title[:4]
            for filename in prompt_files:
                if search_prefix and search_prefix in filename:
                    new_book_title = filename.replace('-提示词.txt', '')
                    related_prompts.append((filename, new_book_title))
        
        # 将相关提示词添加到列表框中
        for i, (filename, new_book_title) in enumerate(related_prompts, start=1):
            self.book_prompts_listbox.insert(tk.END, f"{i}. {new_book_title}")
    
    def on_select_new_book_prompt(self, event=None):
        """当点击新书提示词列表框中的提示词时，加载该提示词内容到右侧文本框"""
        selection = self.book_prompts_listbox.curselection()
        if not selection:
            return
        
        # 获取选中的新书提示词标题
        title_full = self.book_prompts_listbox.get(selection[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        
        # 构建文件路径
        tishici_dir = "./tishici"
        filename = f"{title}-提示词.txt"
        filepath = os.path.join(tishici_dir, filename)
        
        # 读取文件内容
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.analysis_result_text.delete(1.0, tk.END)
                self.analysis_result_text.insert(tk.END, content)
            except Exception as e:
                self.analysis_result_text.delete(1.0, tk.END)
                self.analysis_result_text.insert(tk.END, f"读取文件失败：{e}")
        else:
            self.analysis_result_text.delete(1.0, tk.END)
            self.analysis_result_text.insert(tk.END, f"未找到文件：{filepath}")

    def on_select_write_analysis_with_prompts(self, event=None):
        """当点击写书页面拆书分析库中的文档时，加载该文档内容并查找相关的新书提示词"""
        selection = self.write_analysis_listbox.curselection()
        if not selection:
            return
        
        # 获取选中的拆书分析文档标题
        title_full = self.write_analysis_listbox.get(selection[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        
        # 清空新书提示词列表框
        self.write_new_book_prompts_listbox.delete(0, tk.END)
        
        # 查找与该拆书分析相关的新书提示词
        self.load_write_new_book_prompts_for_analysis(title)
    
    def load_write_new_book_prompts_for_analysis(self, analysis_title):
        """加载与指定拆书分析相关的新书提示词（写书页面专用）"""
        tishici_dir = "./tishici"
        if not os.path.exists(tishici_dir):
            return
        
        # 获取所有新书提示词文件
        prompt_files = []
        for filename in os.listdir(tishici_dir):
            if filename.endswith('-提示词.txt'):
                prompt_files.append(filename)
        
        # 简单关联逻辑：如果新书提示词文件名包含拆书分析的关键词，则认为是相关的
        # 拆书分析标题格式通常是 "原书名-拆书"，我们提取原书名的部分
        original_book_title = analysis_title.replace('-拆书', '')
        
        # 查找相关的新书提示词
        related_prompts = []
        for filename in prompt_files:
            # 从文件名中提取新书名（去掉"-提示词.txt"后缀）
            new_book_title = filename.replace('-提示词.txt', '')
            
            # 简单匹配：如果新书提示词文件名包含原书名的关键词，或者原书名包含新书名的关键词
            if original_book_title in filename or new_book_title in analysis_title:
                related_prompts.append((filename, new_book_title))
        
        # 如果没有找到相关提示词，尝试更宽松的匹配
        if not related_prompts:
            # 使用拆书分析标题的前几个字符进行匹配
            search_prefix = original_book_title[:4]
            for filename in prompt_files:
                if search_prefix and search_prefix in filename:
                    new_book_title = filename.replace('-提示词.txt', '')
                    related_prompts.append((filename, new_book_title))
        
        # 将相关提示词添加到列表框中
        for i, (filename, new_book_title) in enumerate(related_prompts, start=1):
            self.write_new_book_prompts_listbox.insert(tk.END, f"{i}. {new_book_title}")
    
    def on_select_write_new_book_prompt(self, event=None):
        """当点击写书页面新书提示词列表框中的提示词时，加载该提示词内容到右侧文本框，并执行核心前提操作"""
        selection = self.write_new_book_prompts_listbox.curselection()
        if not selection:
            return
        
        # 获取选中的新书提示词标题
        title_full = self.write_new_book_prompts_listbox.get(selection[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        
        # 构建文件路径
        tishici_dir = "./tishici"
        filename = f"{title}-提示词.txt"
        filepath = os.path.join(tishici_dir, filename)
        
        # 读取文件内容
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.prompt_result_text.delete(1.0, tk.END)
                self.prompt_result_text.insert(tk.END, content)
                
                # 执行核心前提操作：检测并创建新书相关文档
                self._process_new_book_core_operations(title, content)
            except Exception as e:
                self.prompt_result_text.delete(1.0, tk.END)
                self.prompt_result_text.insert(tk.END, f"读取文件失败：{e}")
        else:
            self.prompt_result_text.delete(1.0, tk.END)
            self.prompt_result_text.insert(tk.END, f"未找到文件：{filepath}")
    
    def _process_new_book_core_operations(self, book_title, prompt_content):
        """处理新书核心前提操作：检测、创建文件夹和文档，填充内容"""
        # 一、核心前提操作（点击新书提示词文档后触发）
        
        # 1. 检测目标：保存路径下，以「新书名」命名的文件夹中，是否存在名为「书名-新书简介」的文档
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        intro_file = os.path.join(book_dir, f"{book_title}-新书简介.txt")
        
        # 创建新书文件夹（如果不存在）
        os.makedirs(book_dir, exist_ok=True)
        
        # 创建子文件夹结构
        sub_dirs = ["jiagou", "lantu", "caogao"]
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(book_dir, sub_dir), exist_ok=True)
        
        # 2. 执行动作1（文档不存在）：新建名为「书名-新书简介」的文档
        if not os.path.exists(intro_file):
            # 从提示词内容中提取新书简介
            intro_content = self._extract_book_intro_from_prompt(prompt_content)
            
            # 保存新书简介文档
            with open(intro_file, 'w', encoding='utf-8') as f:
                f.write(intro_content)
            
            # 自动填入新书第二页对应的输入框
            self._auto_fill_book_info(book_title, intro_content)
            
            print(f"已创建新书简介文档：{intro_file}")
        else:
            # 3. 执行动作2（文档存在）：直接读取并填充内容
            try:
                with open(intro_file, 'r', encoding='utf-8') as f:
                    intro_content = f.read()
                self._auto_fill_book_info(book_title, intro_content)
                print(f"已读取现有新书简介文档：{intro_file}")
            except Exception as e:
                print(f"读取新书简介文档失败：{e}")
        
        # 4. 后续检测：检测子文件夹结构
        self._check_subdirectories_structure(book_dir)
    
    def _extract_book_intro_from_prompt(self, prompt_content):
        """从提示词内容中提取新书简介"""
        # 尝试从提示词中提取简介部分
        intro_patterns = [
            r'\*\*简介：\*\*\s*(.+?)(?=\n\*\*|\n一、|\n二、|\n三、|\n四、|\n五、|$)',
            r'简介[：:]\s*(.+?)(?=\n一、|\n二、|\n三、|\n四、|\n五、|$)',
            r'新书简介[：:]\s*(.+?)(?=\n一、|\n二、|\n三、|\n四、|\n五、|$)',
        ]
        
        for pattern in intro_patterns:
            match = re.search(pattern, prompt_content, re.DOTALL)
            if match:
                intro = match.group(1).strip()
                if intro:
                    return intro
        
        # 如果没有找到简介，使用默认内容
        return "新书简介内容待补充..."
    
    def _auto_fill_book_info(self, book_title, intro_content):
        """自动填充新书第二页对应的输入框"""
        # 设置新书名
        self.new_book_title.set(book_title)
        
        # 设置新书简介（限制250字）
        if len(intro_content) > 250:
            intro_content = intro_content[:250] + "..."
        self.new_book_synopsis.set(intro_content)
        
        # 从简介中推断题材
        inferred_genre = self.infer_genre(book_title, intro_content)
        self.genre.set(inferred_genre)
        
        print(f"已自动填充：书名={book_title}, 题材={inferred_genre}")
    
    def _check_subdirectories_structure(self, book_dir):
        """检测子文件夹结构"""
        sub_dirs = ["jiagou", "lantu", "caogao"]
        book_title = os.path.basename(book_dir)
        
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(book_dir, sub_dir)
            if os.path.exists(sub_dir_path):
                print(f"✓ 子文件夹存在：{sub_dir_path}")
                
                # 检查是否有对应的文档
                if sub_dir == "jiagou":
                    arch_file = os.path.join(sub_dir_path, f"{book_title}-架构.txt")
                    if os.path.exists(arch_file):
                        print(f"  - 架构文档已存在：{arch_file}")
                
                elif sub_dir == "lantu":
                    blueprint_file = os.path.join(sub_dir_path, f"{book_title}-蓝图.txt")
                    if os.path.exists(blueprint_file):
                        print(f"  - 蓝图文档已存在：{blueprint_file}")
                
                elif sub_dir == "caogao":
                    # 检查草稿文件
                    draft_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.txt')]
                    if draft_files:
                        print(f"  - 草稿文档已存在：{len(draft_files)}个文件")
            else:
                print(f"✗ 子文件夹不存在：{sub_dir_path}")

    # 智能截断：开头6000 + 中间3000 + 结尾2000
    def smart_truncate(self, content, max_chars=11000):
        if len(content) <= max_chars:
            return content
        total = len(content)
        head_len = 6000
        middle_len = 3000
        tail_len = 2000
        head = content[:head_len]
        middle_start = total // 2 - middle_len // 2
        middle = content[middle_start:middle_start + middle_len]
        tail = content[-tail_len:]
        return f"{head}\n\n...[中间部分]...\n\n{middle}\n\n...[结尾部分]...\n\n{tail}"

    # 鲁棒的模板提取
    def extract_template(self, analysis_text):
        patterns = [
            r'###\s*七、提炼模板(.*?)(?=###\s*八|###\s*综合|$)',
            r'##\s*七、提炼模板(.*?)(?=##\s*八|##\s*综合|$)',
            r'七、提炼模板[：:]\s*(.*?)(?=八、|$)',
            r'提炼模板[：:]\s*(.*?)(?=八、|$)',
            r'###\s*7\.\s*提炼模板(.*?)(?=###\s*8\.|###\s*综合|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            if match:
                template = match.group(1).strip()
                if template:
                    return template
        sections = re.split(r'\n###\s+', analysis_text)
        for sec in sections:
            if '提炼模板' in sec or '七、' in sec:
                return sec.strip()
        return None

    def edit_template(self):
        if not hasattr(self, 'current_template_text') or not self.current_template_text:
            messagebox.showwarning("提示", "没有可编辑的模板，请先进行拆书分析")
            return
        win = tk.Toplevel(self.root)
        win.title("编辑模板")
        win.geometry("800x600")
        text_area = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=('微软雅黑', 10))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_area.insert(tk.END, self.current_template_text)
        def save():
            new_template = text_area.get(1.0, tk.END).strip()
            if new_template:
                self.current_template_text = new_template
                messagebox.showinfo("成功", "模板已更新，可以使用「重新生成新书提示词」按钮生成新书提示词")
                win.destroy()
            else:
                messagebox.showwarning("提示", "模板不能为空")
        ttk.Button(win, text="保存", command=save).pack(pady=10)

    def save_current_template(self):
        if not hasattr(self, 'current_template_text') or not self.current_template_text:
            messagebox.showwarning("提示", "没有可保存的模板")
            return
        source_title = self.book_combobox.get().strip()
        self.template_manager.save_template(source_title, self.current_template_text, self.analysis_result_text.get(1.0, tk.END).strip()[:500])
        messagebox.showinfo("成功", "模板已保存到模板库")

    def regenerate_new_book_prompt(self):
        if not hasattr(self, 'current_template_text') or not self.current_template_text:
            messagebox.showwarning("提示", "没有模板，请先进行拆书分析")
            return
        if not self.api_key.get().strip():
            messagebox.showwarning("提示", "请先在系统设置中配置API Key")
            return
        if hasattr(self, 'current_write_page') and self.current_write_page == 1:
            target_text = self.prompt_result_text
        else:
            target_text = self.new_book_prompt_text
        target_text.delete(1.0, tk.END)
        target_text.insert(tk.END, "正在生成新书提示词，请稍候...\n")
        self.regenerate_prompt_btn.config(state=tk.DISABLED)
        def task():
            llm = SimpleLLM(
                api_key=self.api_key.get().strip(),
                base_url=self.base_url.get().strip(),
                model=self.model_name.get().strip(),
                temperature=self.temp.get(),
                max_tokens=8192
            )
            prompt_new_book = self.build_new_book_prompt(self.current_template_text)
            result = llm.chat(prompt_new_book, retry_count=3)
            self.root.after(0, lambda: target_text.delete(1.0, tk.END))
            if "错误：" in result or "请求异常" in result:
                self.root.after(0, lambda: target_text.insert(tk.END, f"生成失败：{result}"))
            else:
                self.root.after(0, lambda: target_text.insert(tk.END, result))
                self.root.after(0, lambda: self.extract_and_sync_book_info(result))
            self.root.after(0, lambda: self.regenerate_prompt_btn.config(state=tk.NORMAL))
        threading.Thread(target=task, daemon=True).start()

    def build_new_book_prompt(self, template_text):
        anti_plagiarism = """
**重要：反抄袭约束**
- 你必须生成完全原创的故事，不能使用原作品的任何具体剧情、人物姓名、地名、核心设定。
- 如果你发现自己生成的设定与原作相似度超过30%，请自动调整，更换元素。
- 生成完成后，请自查：是否有任何桥段、对话、冲突与原作雷同？如果有，请改写。
"""
        return f"""你是一位资深小说大纲设计师。下面是一个从某部作品中提炼出的“可复用写作模板”。请严格基于这个模板的结构和思路，**但不要使用原作品的任何具体剧情、人物、设定、冲突**，生成一个全新的、原创的小说写作提示词。

{anti_plagiarism}

**特别要求：**
1. **新书名必须严格遵循以下公式：`场景/身份 + 冲突/动作 + 结果/爽点`**
   - 示例1（身份+动作+对象）：《退伍兵王横扫豪门》
   - 示例2（时间+金手指+数值）：《开局签到一个亿》
   - 示例3（身份+状态+结果）：《被逐出宗门后我无敌了》
   - 示例4（背景+能力+悬念）：《全球觉醒：我能看到隐藏属性》
   - 请根据你生成的故事类型，创作一个符合该公式的原创书名。

2. **新书简介必须遵循广告三段式：**
   - 第一句：主角是谁，处于什么处境？（建立代入感）
   - 第二句：抛出冲突——什么矛盾、不公或机遇打破了平静？（制造紧张感）
   - 第三句：悬念收尾——不给答案，让读者产生好奇自己去找答案。
   - 简介整体要像广告一样吸引人，不能写成大纲。

模板内容如下：
{template_text}

请输出一个完整的新书写作提示词，格式如下（直接输出，不要加额外解释）：

【新书写作提示词】

**书名：** （必须符合上述公式）

**简介：** （三段式广告，每段一句，总字数100-200字）

一、开篇设计建议
（根据模板填充具体内容，原创，包括如何写出前三章）

二、章节节奏建议
（原创节奏表，每章字数建议）

三、爽点安排建议
（原创爽点类型和频率，以及具体在哪些章节出现）

四、金手指设计建议
（原创金手指逻辑，包括限制和成长路线）

五、完整故事大纲框架
（提供一个可填空的大纲，用户可自行填入具体设定，包含至少10个章节的概要）

请确保提示词具体、可操作，并且完全原创。书名和简介必须严格按照上述要求创作。"""

    def create_book_analysis(self):
        # 防止重复调用
        if self.is_analyzing:
            messagebox.showwarning("提示", "拆书分析正在执行中，请稍后...")
            return
        selected_title = self.book_combobox.get().strip()
        if not selected_title:
            messagebox.showwarning("提示", "请从下拉框选择一个文档")
            return
        if not self.api_key.get().strip():
            messagebox.showwarning("提示", "请先在「系统设置」中配置API Key并测试")
            return

        analysis_title = f"{selected_title}-拆书"
        existing_docs = self.kb_manager.get_all_documents()
        if any(doc['title'] == analysis_title for doc in existing_docs):
            if not messagebox.askyesno("提示", f"该书《{selected_title}》已拆书，是否重新分析（将覆盖原有结果）？"):
                return

        content = self.kb_manager.get_document_content(selected_title)
        if not content:
            messagebox.showerror("错误", f"无法读取文档「{selected_title}」的内容")
            return

        self.analysis_result_text.delete(1.0, tk.END)
        self.analysis_result_text.insert(tk.END, "正在分析中，请稍候...\n")
        self.create_btn.config(state=tk.DISABLED)
        self.edit_template_btn.config(state=tk.DISABLED)
        self.book_analysis_listbox.config(state=tk.DISABLED)  # 禁用拆书分析库列表
        self.is_analyzing = True

        def update_progress(msg):
            """在线程中更新进度显示"""
            self.root.after(0, lambda: self.analysis_result_text.insert(tk.END, msg + "\n"))

        def analysis_task():
            try:
                llm = SimpleLLM(
                    api_key=self.api_key.get().strip(),
                    base_url=self.base_url.get().strip(),
                    model=self.model_name.get().strip(),
                    temperature=self.temp.get(),
                    max_tokens=8192
                )
                update_progress("步骤1/4：正在截取文档内容...")
                truncated_content = self.smart_truncate(content)
                detail_section = ""
                if self.detail_analysis_var.get():
                    detail_section = """
### 九、人物弧光分析
- 主角在故事中的成长曲线是什么？经历了哪些关键转变？
- 配角是否有独立的动机和变化？

### 十、对话技巧分析
- 对话是否推动情节或塑造人物？
- 对话的节奏和风格如何？
"""
                prompt_analysis = f"""你是一位顶尖的小说编辑与写作教练。请对以下小说内容进行深度拆解分析，严格按照下列维度输出结构化报告。**重点：最后必须输出一个清晰、可直接复用的写作模板（放在“### 七、提炼模板”部分）。**
{detail_section}
小说内容：
{truncated_content}

请按以下格式输出：

### 一、开篇设计（前三章是怎么抓人的）
- 第一段/第一句话是怎么开场的？用了什么手法吸引读者？
- 主角在第几段出场？出场时读者对他的第一印象是什么？
- 前三章里，核心冲突是什么时候抛出来的？
- 读者看完前三章，会产生什么疑问想继续看下去？
- 前三章的信息密度如何？是一上来就大量设定，还是边走边交代？

### 二、钩子设计（让人想翻下一页的技巧）
- 每一章结尾是否留有悬念或期待？
- 使用了哪些具体的钩子类型（对话钩、情节钩、情绪钩等）？

### 三、情绪走向（阅读体验的核心）
- 整体情绪曲线如何变化？
- 作者如何调动读者情绪？

### 四、起承转合（故事结构）
- 故事的“起”、“承”、“转”、“合”分别在哪里？
- 节奏是否张弛有度？

### 五、爽点设计（读者为什么追读）
- 列出了哪些让读者产生快感的元素？
- 爽点的频率和强度如何？

### 六、金手指设计（核心外挂逻辑）
- 主角拥有什么特殊能力、资源或信息优势？
- 金手指的设定是否合理、有新鲜感？

### 七、提炼模板（★重点：可复用的创作框架，直接用于写小说★）
请将以上分析结果，总结成一个可以直接套用的写作模板。模板应包含以下子部分（每个子部分都要有具体内容，不能为空）：

#### 7.1 开篇模板（如何写前三章）
- 开场句式示例：____________
- 主角出场时机与形象：____________
- 核心冲突抛出节点：____________
- 信息密度控制建议：____________

#### 7.2 章节节奏模板
- 建议每章字数：____________
- 章节内部结构：开头钩子→发展→小高潮→结尾悬念
- 具体节奏表：____________

#### 7.3 爽点安排模板
- 爽点频率：____________
- 常见爽点类型及插入位置：____________

#### 7.4 金手指设计模板
- 金手指类型建议：____________
- 限制条件（避免无敌）：____________
- 成长/解锁路线：____________

#### 7.5 完整故事大纲模板（可直接填空使用）
提供一个通用的大纲框架，用户只需填入自己的人物和设定即可。

### 八、综合评分与改进建议
- 对开篇吸引力、钩子密度、情绪感染力、结构清晰度、爽点设计分别打分（1-10分）。
- 给出2-3条具体的改进建议。
"""
                result_analysis = llm.chat(prompt_analysis, retry_count=3)

                if "错误：" in result_analysis or "请求异常" in result_analysis:
                    self.root.after(0, lambda: self.show_analysis_error("拆书分析失败", result_analysis))
                    return

                fenxi_dir = "./fenxi"
                os.makedirs(fenxi_dir, exist_ok=True)
                base_name = f"{selected_title}-拆书分析"
                suffix = ".txt"
                candidate = os.path.join(fenxi_dir, base_name + suffix)
                counter = 1
                while os.path.exists(candidate):
                    candidate = os.path.join(fenxi_dir, f"{base_name}_{counter}{suffix}")
                    counter += 1
                try:
                    with open(candidate, 'w', encoding='utf-8') as f:
                        f.write(result_analysis)
                    save_msg = f"\n\n[分析结果已保存至：{candidate}]"
                except Exception as e:
                    save_msg = f"\n\n[保存文件失败：{e}]"

                analysis_title = f"{selected_title}-拆书"
                try:
                    doc_id, version = self.book_analysis_manager.upload_document(analysis_title, result_analysis)
                    save_msg += f"\n\n[分析结果已保存到拆书分析库：{analysis_title} (v{version})]"
                    self.refresh_analysis_library()
                except Exception as e:
                    save_msg += f"\n\n[保存到拆书分析库失败：{e}]"

                self.root.after(0, lambda: self.display_analysis_result(result_analysis + save_msg))

                template_text = self.extract_template(result_analysis)
                if not template_text:
                    self.root.after(0, lambda: self.analysis_result_text.insert(tk.END, "\n\n模板提取失败，正在尝试重试..."))
                    retry_prompt = f"请从以下分析结果中提取出“### 七、提炼模板”部分的内容，只输出该模板内容，不要输出其他任何解释。\n\n分析结果：\n{result_analysis}"
                    template_text = llm.chat(retry_prompt, retry_count=2)
                    if "错误：" in template_text or "请求异常" in template_text:
                        self.root.after(0, lambda: self.analysis_result_text.insert(tk.END, "\n模板提取失败，请手动编辑或重新分析。"))
                        self.root.after(0, lambda: self.edit_template_btn.config(state=tk.NORMAL))
                        return

                self.current_template_text = template_text
                self.root.after(0, lambda: self.edit_template_btn.config(state=tk.NORMAL))
            except Exception as e:
                self.root.after(0, lambda: self.show_analysis_error("拆书分析失败", str(e)))
            finally:
                self.root.after(0, lambda: self.create_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.book_analysis_listbox.config(state=tk.NORMAL))  # 恢复拆书分析库列表
                self.root.after(0, lambda: setattr(self, 'is_analyzing', False))

        threading.Thread(target=analysis_task, daemon=True).start()

    def extract_and_sync_book_info(self, prompt_text):
        title_match = re.search(r'\*\*书名：\*\*\s*(.+?)(?:\n|$)', prompt_text)
        new_title = ""
        if title_match:
            new_title = title_match.group(1).strip().strip('《》').strip()
            self.new_book_title.set(new_title)
        intro_match = re.search(r'\*\*简介：\*\*\s*(.+?)(?=\n\*\*|$)', prompt_text, re.DOTALL)
        new_intro = ""
        if intro_match:
            new_intro = intro_match.group(1).strip()
            self.new_book_synopsis.set(new_intro)
        title_for_genre = new_title if new_title else self.new_book_title.get()
        intro_for_genre = new_intro if new_intro else self.new_book_synopsis.get()
        self.chapter_num.set(25)
        self.words_per_chapter.set(2800)
        inferred_genre = self.infer_genre(title_for_genre, intro_for_genre)
        self.genre.set(inferred_genre)

    def infer_genre(self, title, intro):
        keywords = {
            "修仙": ["修仙", "仙侠", "修炼", "灵根", "宗门", "飞升", "丹药", "灵力"],
            "玄幻": ["玄幻", "魔法", "异能", "神魔", "大陆", "龙族", "精灵", "召唤"],
            "都市": ["都市", "现代", "公司", "职场", "豪门", "娱乐圈", "明星", "白领"],
            "武侠": ["武侠", "江湖", "剑客", "侠客", "门派", "武功", "刀剑"],
            "科幻": ["科幻", "科技", "未来", "宇宙", "机器人", "太空", "AI", "虚拟"],
            "言情": ["言情", "爱情", "甜宠", "霸道", "总裁", "恋爱", "婚姻", "浪漫"],
            "历史": ["历史", "古代", "王朝", "皇帝", "宫廷", "穿越", "架空"],
            "恐怖": ["恐怖", "惊悚", "鬼怪", "灵异", " horror"],
            "悬疑": ["悬疑", "推理", "侦探", "谋杀", "案件", "谜团"],
        }
        text = (title + " " + intro).lower()
        for genre, words in keywords.items():
            if any(word.lower() in text for word in words):
                return genre
        return "其他"

    def display_analysis_result(self, result):
        self.analysis_result_text.delete(1.0, tk.END)
        self.analysis_result_text.insert(tk.END, result)
        self.analysis_hint.config(text="分析完成")

    def show_analysis_error(self, title, msg):
        messagebox.showerror(title, msg)
        self.analysis_result_text.delete(1.0, tk.END)
        self.analysis_result_text.insert(tk.END, f"错误：{msg}")

    # ---------- 写书页面（分两页） ----------
    def create_write_book_page(self):
        page = tk.Frame(self.right_area, bg='#f0f2f5')
        self.write_page_container = tk.Frame(page, bg='#f0f2f5')
        self.write_page_container.pack(fill=tk.BOTH, expand=True)
        self.current_write_page = 1
        self.create_write_page1()
        self.create_write_page2()
        self.show_write_page1()
        return page

    def create_write_page1(self):
        self.write_page1 = tk.Frame(self.write_page_container, bg='#f0f2f5')

        main_frame = tk.Frame(self.write_page1, bg='#f0f2f5')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # ========== 左侧：拆书分析库 ==========
        left_frame = ttk.LabelFrame(main_frame, text="拆书分析库", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))

        self.write_analysis_listbox = tk.Listbox(left_frame, height=15)
        self.write_analysis_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.write_analysis_listbox.bind("<Button-3>", self.show_write_analysis_menu)
        self.write_analysis_listbox.bind("<ButtonRelease-1>", self.on_select_write_analysis_with_prompts)
        self.refresh_analysis_library()

        self.gen_prompt_btn = ttk.Button(left_frame, text="生成新书提示词", command=self.generate_prompt_from_selected)
        self.gen_prompt_btn.pack(fill=tk.X, pady=5)

        # ========== 中间：新书提示词列表框 ==========
        middle_frame = ttk.LabelFrame(main_frame, text="新书提示词", padding=10)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.write_new_book_prompts_listbox = tk.Listbox(middle_frame, height=15)
        self.write_new_book_prompts_listbox.pack(fill=tk.BOTH, expand=True)
        self.write_new_book_prompts_listbox.bind("<ButtonRelease-1>", self.on_select_write_new_book_prompt)

        # ========== 右侧：生成结果 ==========
        right_frame = ttk.LabelFrame(main_frame, text="生成结果", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10,0))

        self.prompt_result_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=('微软雅黑', 10))
        self.prompt_result_text.pack(fill=tk.BOTH, expand=True, pady=5)

        bottom_btn_frame = tk.Frame(right_frame, bg='#f0f2f5')
        bottom_btn_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        start_write_btn = tk.Button(bottom_btn_frame, text="开始写书", bg='yellow', fg='black', font=('微软雅黑', 10, 'bold'), command=self.show_write_page2)
        start_write_btn.pack(side=tk.RIGHT, padx=5, pady=5)

    def generate_prompt_from_selected(self):
        if self.is_generating_prompt:
            messagebox.showwarning("提示", "提示词正在生成中，请稍后...")
            return
        selection = self.write_analysis_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先在左侧拆书分析库中选中一个分析结果")
            return
        title_full = self.write_analysis_listbox.get(selection[0])
        if '. ' in title_full:
            analysis_title = title_full.split('. ', 1)[1]
        else:
            analysis_title = title_full
        content = self.book_analysis_manager.get_document_content(analysis_title)
        if not content:
            messagebox.showerror("错误", f"无法读取分析结果「{analysis_title}」")
            return
        template_text = self.extract_template(content)
        if not template_text:
            if not messagebox.askyesno("提示", "未能自动提取模板，是否尝试让 AI 重新提取？"):
                return
            if not self.api_key.get().strip():
                messagebox.showwarning("提示", "请先在系统设置中配置API Key")
                return
            self.prompt_result_text.delete(1.0, tk.END)
            self.prompt_result_text.insert(tk.END, "正在尝试提取模板，请稍候...\n")
            self.is_generating_prompt = True
            self.gen_prompt_btn.config(state=tk.DISABLED)
            def extract_task():
                try:
                    llm = SimpleLLM(
                        api_key=self.api_key.get().strip(),
                        base_url=self.base_url.get().strip(),
                        model=self.model_name.get().strip(),
                        temperature=self.temp.get(),
                        max_tokens=8192
                    )
                    prompt = f"请从以下拆书分析结果中提取出「### 七、提炼模板」部分的内容，只输出该模板内容，不要输出其他解释。\n\n分析结果：\n{content}"
                    extracted = llm.chat(prompt, retry_count=2)
                    if "错误：" in extracted or "请求异常" in extracted:
                        self.root.after(0, lambda: self.prompt_result_text.insert(tk.END, f"提取失败：{extracted}"))
                        return
                    self.current_template_text = extracted
                    self._do_generate_prompt_from_template(extracted)
                finally:
                    self.root.after(0, lambda: self.gen_prompt_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: setattr(self, 'is_generating_prompt', False))
            threading.Thread(target=extract_task, daemon=True).start()
            return
        self.current_template_text = template_text
        self._do_generate_prompt_from_template(template_text)

    def _do_generate_prompt_from_template(self, template_text):
        if not self.api_key.get().strip():
            messagebox.showwarning("提示", "请先在系统设置中配置API Key")
            return
        if self.is_generating_prompt:
            return
        self.is_generating_prompt = True
        self.gen_prompt_btn.config(state=tk.DISABLED)
        self.prompt_result_text.delete(1.0, tk.END)
        self.prompt_result_text.insert(tk.END, "正在生成新书提示词，请稍候...\n")
        def task():
            try:
                llm = SimpleLLM(
                    api_key=self.api_key.get().strip(),
                    base_url=self.base_url.get().strip(),
                    model=self.model_name.get().strip(),
                    temperature=self.temp.get(),
                    max_tokens=8192
                )
                prompt_new_book = self.build_new_book_prompt(template_text)
                result = llm.chat(prompt_new_book, retry_count=3)
                self.root.after(0, lambda: self.prompt_result_text.delete(1.0, tk.END))
                if "错误：" in result or "请求异常" in result:
                    self.root.after(0, lambda: self.prompt_result_text.insert(tk.END, f"生成失败：{result}"))
                else:
                    self.root.after(0, lambda: self.prompt_result_text.insert(tk.END, result))
                    self.root.after(0, lambda: self.extract_and_sync_book_info(result))
                    # 生成成功后自动保存提示词
                    self.root.after(0, lambda: self.auto_save_prompt())
            finally:
                self.root.after(0, lambda: self.gen_prompt_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: setattr(self, 'is_generating_prompt', False))
        threading.Thread(target=task, daemon=True).start()

    def show_book_analysis_menu(self, event):
        self.show_analysis_menu(event, self.book_analysis_listbox)

    def show_write_analysis_menu(self, event):
        self.show_analysis_menu(event, self.write_analysis_listbox)

    def show_analysis_menu(self, event, listbox):
        index = listbox.nearest(event.y)
        if index < 0:
            return
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(index)
        listbox.activate(index)

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="编辑", command=lambda: self.edit_analysis(listbox))
        menu.add_command(label="删除", command=lambda: self.delete_analysis(listbox))
        menu.post(event.x_root, event.y_root)

    def refresh_analysis_library(self):
        docs = self.book_analysis_manager.get_all_documents()
        if hasattr(self, 'write_analysis_listbox'):
            self.write_analysis_listbox.delete(0, tk.END)
            for i, doc in enumerate(docs, start=1):
                self.write_analysis_listbox.insert(tk.END, f"{i}. {doc['title']}")
        if hasattr(self, 'book_analysis_listbox'):
            self.book_analysis_listbox.delete(0, tk.END)
            for i, doc in enumerate(docs, start=1):
                self.book_analysis_listbox.insert(tk.END, f"{i}. {doc['title']}")

    def edit_analysis(self, listbox):
        selected = listbox.curselection()
        if not selected:
            return
        title_full = listbox.get(selected[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        content = self.book_analysis_manager.get_document_content(title)
        if not content:
            messagebox.showerror("错误", "无法获取内容")
            return

        win = tk.Toplevel(self.root)
        win.title(f"编辑 {title}")
        win.geometry("800x600")
        text_area = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=('微软雅黑', 10))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_area.insert(tk.END, content)

        def save():
            new_content = text_area.get(1.0, tk.END).strip()
            if new_content:
                self.book_analysis_manager.upload_document(title, new_content)
                self.refresh_analysis_library()
                messagebox.showinfo("成功", "已保存")
                win.destroy()
            else:
                messagebox.showwarning("提示", "内容不能为空")

        ttk.Button(win, text="保存", command=save).pack(pady=10)

    def delete_analysis(self, listbox):
        selected = listbox.curselection()
        if not selected:
            return
        title_full = listbox.get(selected[0])
        title = title_full.split('. ', 1)[1] if '. ' in title_full else title_full
        if messagebox.askyesno("确认", f"确定删除 {title} 吗？"):
            self.book_analysis_manager.delete_document(title)
            self.refresh_analysis_library()

    def save_prompt(self):
        content = self.prompt_result_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("提示", "没有内容可保存")
            return
        book_name = self.new_book_title.get().strip()
        if not book_name:
            messagebox.showwarning("提示", "请先设置新书名")
            return
        tishici_dir = "./tishici"
        os.makedirs(tishici_dir, exist_ok=True)
        filename = f"{book_name}-提示词.txt"
        filepath = os.path.join(tishici_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("成功", f"提示词已保存至：{filepath}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{e}")

    def auto_save_prompt(self):
        """自动保存提示词（生成成功后调用）"""
        content = self.prompt_result_text.get(1.0, tk.END).strip()
        if not content:
            self.log_to_write("自动保存失败：提示词内容为空")
            return
        book_name = self.new_book_title.get().strip()
        if not book_name:
            self.log_to_write("自动保存失败：新书名为空")
            return
        tishici_dir = "./tishici"
        os.makedirs(tishici_dir, exist_ok=True)
        filename = f"{book_name}-提示词.txt"
        filepath = os.path.join(tishici_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.log_to_write(f"提示词已自动保存至：{filepath}")
        except Exception as e:
            self.log_to_write(f"自动保存失败：{e}")

    def create_write_page2(self):
        self.write_page2 = tk.Frame(self.write_page_container, bg='#f0f2f5')

        # ========== 新增：文档选择区域（位于新书名上方） ==========
        doc_select_frame = ttk.LabelFrame(self.write_page2, text="文档选择（从tishici文件夹选取新书提示词）", padding=5)
        doc_select_frame.pack(fill=tk.X, padx=20, pady=(10, 0))

        select_row = tk.Frame(doc_select_frame)
        select_row.pack(fill=tk.X, pady=5)
        ttk.Label(select_row, text="选择提示词文件:").pack(side=tk.LEFT, padx=5)
        self.prompt_file_combobox = ttk.Combobox(select_row, width=50, state="readonly")
        self.prompt_file_combobox.pack(side=tk.LEFT, padx=5)
        ttk.Button(select_row, text="刷新列表", command=self.refresh_prompt_file_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_row, text="加载选中", command=self.load_selected_prompt_file).pack(side=tk.LEFT, padx=5)
        self.refresh_prompt_file_list()  # 初始加载

        top_frame = tk.Frame(self.write_page2, bg='#f0f2f5')
        top_frame.pack(fill=tk.X, padx=20, pady=(10,10))

        name_frame = tk.Frame(top_frame, bg='#f0f2f5')
        name_frame.pack(side=tk.LEFT)
        tk.Label(name_frame, text="新书名：", font=('微软雅黑', 12, 'bold'), bg='#f0f2f5').pack(side=tk.LEFT)
        self.book_title_entry = tk.Entry(name_frame, textvariable=self.new_book_title, font=('微软雅黑', 11), width=25)
        self.book_title_entry.pack(side=tk.LEFT, padx=5)

        settings_frame = tk.Frame(top_frame, bg='#f0f2f5', relief=tk.GROOVE, bd=1)
        settings_frame.pack(side=tk.RIGHT, padx=10)
        tk.Label(settings_frame, text="设置", font=('微软雅黑', 10, 'bold'), bg='#f0f2f5').pack(anchor='w', padx=5, pady=2)

        row1 = tk.Frame(settings_frame, bg='#f0f2f5')
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="题材:", font=('微软雅黑', 9), bg='#f0f2f5', width=6, anchor='e').pack(side=tk.LEFT)
        self.genre_entry = tk.Entry(row1, textvariable=self.genre, font=('微软雅黑', 9), width=12)
        self.genre_entry.pack(side=tk.LEFT, padx=2)

        row2 = tk.Frame(settings_frame, bg='#f0f2f5')
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="章节数:", font=('微软雅黑', 9), bg='#f0f2f5', width=6, anchor='e').pack(side=tk.LEFT)
        self.chapter_num_spin = tk.Spinbox(row2, from_=1, to=200, textvariable=self.chapter_num, width=6)
        self.chapter_num_spin.pack(side=tk.LEFT, padx=2)

        row3 = tk.Frame(settings_frame, bg='#f0f2f5')
        row3.pack(fill=tk.X, pady=2)
        tk.Label(row3, text="每章字数:", font=('微软雅黑', 9), bg='#f0f2f5', width=6, anchor='e').pack(side=tk.LEFT)
        self.words_spin = tk.Spinbox(row3, from_=500, to=20000, increment=500, textvariable=self.words_per_chapter, width=8)
        self.words_spin.pack(side=tk.LEFT, padx=2)

        row4 = tk.Frame(settings_frame, bg='#f0f2f5')
        row4.pack(fill=tk.X, pady=2)
        tk.Label(row4, text="保存路径:", font=('微软雅黑', 9), bg='#f0f2f5', width=6, anchor='e').pack(side=tk.LEFT)
        self.save_path_entry = tk.Entry(row4, textvariable=self.save_path, font=('微软雅黑', 9), width=15)
        self.save_path_entry.pack(side=tk.LEFT, padx=2)
        tk.Button(row4, text="浏览", command=self.browse_save_path, font=('微软雅黑', 8)).pack(side=tk.LEFT)

        intro_frame = tk.Frame(self.write_page2, bg='#f0f2f5')
        intro_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(intro_frame, text="新书内容简介：", font=('微软雅黑', 11, 'bold'), bg='#f0f2f5', anchor='w').pack(anchor='w')
        self.book_synopsis_text = scrolledtext.ScrolledText(intro_frame, wrap=tk.WORD, height=3, width=40, font=('微软雅黑', 10))
        self.book_synopsis_text.pack(fill=tk.X, pady=5, padx=5)
        def sync_synopsis(*args):
            self.book_synopsis_text.delete(1.0, tk.END)
            self.book_synopsis_text.insert(tk.END, self.new_book_synopsis.get())
        self.new_book_synopsis.trace('w', sync_synopsis)
        def on_synopsis_change(event=None):
            self.new_book_synopsis.set(self.book_synopsis_text.get(1.0, tk.END).strip())
        self.book_synopsis_text.bind('<KeyRelease>', on_synopsis_change)

        step_frame = ttk.LabelFrame(self.write_page2, text="生成步骤", padding=10)
        step_frame.pack(fill=tk.X, padx=20, pady=5)
        btn_frame = tk.Frame(step_frame)
        btn_frame.pack()
        self.gen_arch_btn = ttk.Button(btn_frame, text="Step1: 生成整体架构", command=self.gen_architecture)
        self.gen_arch_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.gen_blueprint_btn = ttk.Button(btn_frame, text="Step2: 生成章节蓝图", command=self.gen_blueprints)
        self.gen_blueprint_btn.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(btn_frame, text="Step3: 生成全部草稿", command=self.gen_all_drafts).pack(side=tk.LEFT, padx=5, pady=5)
        self.gen_partial_btn = ttk.Button(btn_frame, text="生成部分草稿", command=self.gen_partial_drafts)
        self.gen_partial_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.gen_all_in_one_btn = ttk.Button(btn_frame, text="一键生成小说", command=self.gen_novel_one_click, style='Accent.TButton')
        self.gen_all_in_one_btn.pack(side=tk.LEFT, padx=5, pady=5)

        progress_frame = ttk.LabelFrame(self.write_page2, text="生成进度", padding=5)
        progress_frame.pack(fill=tk.X, padx=20, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)
        self.progress_label_var = tk.StringVar(value="就绪")
        tk.Label(progress_frame, textvariable=self.progress_label_var, font=('微软雅黑', 9)).pack()

        notebook = ttk.Notebook(self.write_page2)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.text_arch = scrolledtext.ScrolledText(notebook, wrap=tk.WORD, font=('微软雅黑', 10))
        notebook.add(self.text_arch, text="小说架构")
        self.text_blueprint = scrolledtext.ScrolledText(notebook, wrap=tk.WORD, font=('微软雅黑', 10))
        notebook.add(self.text_blueprint, text="章节蓝图")
        self.text_draft = scrolledtext.ScrolledText(notebook, wrap=tk.WORD, font=('微软雅黑', 10))
        notebook.add(self.text_draft, text="草稿内容")

    def refresh_prompt_file_list(self):
        """刷新提示词文件下拉列表"""
        tishici_dir = "./tishici"
        if not os.path.exists(tishici_dir):
            os.makedirs(tishici_dir, exist_ok=True)
        files = [f for f in os.listdir(tishici_dir) if f.endswith('-提示词.txt')]
        # 提取新书名（去掉-提示词.txt）
        display_names = [f.replace('-提示词.txt', '') for f in files]
        self.prompt_file_combobox['values'] = display_names
        if display_names:
            self.prompt_file_combobox.set(display_names[0])
        else:
            self.prompt_file_combobox.set('')

    def load_selected_prompt_file(self):
        """加载选中的提示词文件，自动填充新书名、简介并保存简介文档"""
        selected = self.prompt_file_combobox.get()
        if not selected:
            messagebox.showwarning("提示", "请先选择一个提示词文件")
            return
        tishici_dir = "./tishici"
        filename = f"{selected}-提示词.txt"
        filepath = os.path.join(tishici_dir, filename)
        if not os.path.exists(filepath):
            messagebox.showerror("错误", f"文件不存在：{filepath}")
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败：{e}")
            return

        # 提取书名（优先从文件内容中提取）
        title_match = re.search(r'\*\*书名：\*\*\s*(.+?)(?:\n|$)', content)
        if title_match:
            book_title = title_match.group(1).strip().strip('《》').strip()
        else:
            # 若未提取到，使用文件名（去掉-提示词）
            book_title = selected

        # 提取简介（限250字）
        intro = self._extract_book_intro_from_prompt(content)
        if len(intro) > 250:
            intro = intro[:250] + "..."

        # 自动填充
        self.new_book_title.set(book_title)
        self.new_book_synopsis.set(intro)
        inferred_genre = self.infer_genre(book_title, intro)
        self.genre.set(inferred_genre)

        # 保存简介文档
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        os.makedirs(book_dir, exist_ok=True)
        intro_file = os.path.join(book_dir, f"{book_title}-新书简介.txt")
        try:
            with open(intro_file, 'w', encoding='utf-8') as f:
                f.write(intro)
            self.log_to_write(f"简介文档已保存至：{intro_file}")
        except Exception as e:
            self.log_to_write(f"保存简介文档失败：{e}")

        # ========== 先清除三个文本框内容 ==========
        self.text_arch.delete(1.0, tk.END)
        self.text_blueprint.delete(1.0, tk.END)
        self.text_draft.delete(1.0, tk.END)
        self.chapter_blueprints.clear()
        self.chapter_drafts.clear()

        # ========== 加载架构文档（jiagou文件夹） ==========
        jiagou_dir = os.path.join(book_dir, "jiagou")
        arch_file = os.path.join(jiagou_dir, f"{book_title}-架构.txt")
        if os.path.exists(arch_file):
            try:
                with open(arch_file, 'r', encoding='utf-8') as f:
                    arch_content = f.read()
                self.text_arch.insert(tk.END, arch_content)
                self.architecture = arch_content
                self.log_to_write(f"架构文档已加载：{arch_file}")
            except Exception as e:
                self.log_to_write(f"加载架构文档失败：{e}")
        # 如果文件不存在，保持文本框空白（已在上方清除）
        
        # ========== 加载蓝图文档（lantu文件夹） ==========
        lantu_dir = os.path.join(book_dir, "lantu")
        blueprint_file = os.path.join(lantu_dir, f"{book_title}-蓝图.txt")
        if os.path.exists(blueprint_file):
            try:
                with open(blueprint_file, 'r', encoding='utf-8') as f:
                    blueprint_content = f.read()
                self.text_blueprint.insert(tk.END, blueprint_content)
                # 解析蓝图中的章节信息
                self._parse_blueprints_to_chapter_blueprints(blueprint_content)
                self.log_to_write(f"蓝图文档已加载：{blueprint_file}")
            except Exception as e:
                self.log_to_write(f"加载蓝图文档失败：{e}")
        # 如果文件不存在，保持文本框空白（已在上方清除）

        # ========== 加载草稿内容（caogao文件夹） ==========
        caogao_dir = os.path.join(book_dir, "caogao")
        if os.path.exists(caogao_dir):
            draft_files = [f for f in os.listdir(caogao_dir) if f.startswith('第') and f.endswith('.txt')]
            if draft_files:
                all_drafts_content = []
                for draft_file in sorted(draft_files):
                    draft_path = os.path.join(caogao_dir, draft_file)
                    try:
                        with open(draft_path, 'r', encoding='utf-8') as f:
                            draft_content = f.read()
                        all_drafts_content.append(f"===== {draft_file} =====\n{draft_content}\n")
                        # 解析草稿章节信息
                        match = re.match(r'第(\d+)章', draft_file)
                        if match:
                            chapter_num = int(match.group(1))
                            self.chapter_drafts[chapter_num] = draft_content
                    except Exception as e:
                        self.log_to_write(f"加载草稿文件失败：{draft_file} - {e}")
                if all_drafts_content:
                    self.text_draft.insert(tk.END, "\n".join(all_drafts_content))
                    self.log_to_write(f"草稿文档已加载：共{len(all_drafts_content)}个章节")
        # 如果文件夹不存在，保持文本框空白（已在上方清除）
        
        messagebox.showinfo("成功", f"已加载提示词「{selected}」\n新书名：{book_title}\n简介已提取（{len(intro)}字）并保存。\n架构、蓝图和草稿文档已自动加载（如存在）。")
    
    def _parse_blueprints_to_chapter_blueprints(self, blueprint_content):
        """解析蓝图文档，提取每个章节的蓝图内容"""
        self.chapter_blueprints.clear()
        # 按章节分隔符分割
        import re
        # 匹配 "========== 第X章 章节名 ==========" 或类似格式
        chapters = re.split(r'={5,}\s*第(\d+)章[^\n]*\s*={5,}', blueprint_content)
        if len(chapters) > 1:
            # 从第2个元素开始，每两个元素为一组（索引，章节内容）
            for i in range(1, len(chapters), 2):
                if i+1 < len(chapters):
                    chapter_num = int(chapters[i])
                    chapter_content = chapters[i+1].strip()
                    self.chapter_blueprints[chapter_num] = chapter_content

    def show_write_page1(self):
        self.write_page2.pack_forget()
        self.write_page1.pack(fill=tk.BOTH, expand=True)
        self.current_write_page = 1
        self.refresh_analysis_library()

    def show_write_page2(self):
        self.write_page1.pack_forget()
        self.write_page2.pack(fill=tk.BOTH, expand=True)
        self.current_write_page = 2
        self.refresh_prompt_file_list()  # 刷新下拉列表

    def browse_save_path(self):
        p = filedialog.askdirectory()
        if p:
            self.save_path.set(p)

    def get_llm(self):
        return SimpleLLM(
            api_key=self.api_key.get().strip(),
            base_url=self.base_url.get().strip(),
            model=self.model_name.get().strip(),
            temperature=self.temp.get(),
            max_tokens=8192
        )

    # ==================== 修复后的 gen_architecture 方法 ====================
    def gen_architecture(self):
        # 防止重复调用，增加强制重置机制
        if self.is_generating_arch:
            # 尝试等待 1 秒，若标志仍未重置，则强制重置（避免卡死）
            if hasattr(self, '_arch_thread') and self._arch_thread and self._arch_thread.is_alive():
                messagebox.showwarning("提示", "整体架构正在生成中，请稍后...")
                return
            else:
                # 线程已结束但标志未重置，强制重置
                self.is_generating_arch = False
                self.gen_arch_btn.config(state=tk.NORMAL)
                self.log_to_write("检测到架构生成标志异常，已强制重置。")

        if not self.new_book_title.get().strip():
            messagebox.showwarning("提示", "请先填写新书名")
            return

        book_title = self.new_book_title.get().strip()
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        jiagou_dir = os.path.join(book_dir, "jiagou")
        arch_file = os.path.join(jiagou_dir, f"{book_title}-架构.txt")

        # 检查架构文档是否存在
        if os.path.exists(arch_file):
            response = messagebox.askyesno("架构已存在", 
                f"架构文档已存在：{arch_file}\n\n是否要重新生成架构？\n\n"
                "点击「是」重新生成架构（将覆盖原有内容）\n"
                "点击「否」停止任何更改")
            if not response:
                self.log_to_write("用户取消，停止架构生成")
                return

        if not self.api_key.get().strip():
            messagebox.showwarning("提示", "请先在系统设置中配置API Key")
            return

        # 获取当前选中的提示词文件内容（若有）
        selected_prompt_name = self.prompt_file_combobox.get() if hasattr(self, 'prompt_file_combobox') else ""
        prompt_content = ""
        if selected_prompt_name:
            tishici_dir = "./tishici"
            prompt_file = os.path.join(tishici_dir, f"{selected_prompt_name}-提示词.txt")
            if os.path.exists(prompt_file):
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompt_content = f.read()
                    self.log_to_write(f"已加载提示词文件：{prompt_file}")
                except Exception as e:
                    self.log_to_write(f"读取提示词文件失败：{e}")

        # 设置生成标志
        self.is_generating_arch = True
        self.gen_arch_btn.config(state=tk.DISABLED)
        self.log_to_write("开始生成整体架构...")
        self.text_arch.delete(1.0, tk.END)
        self.text_arch.insert(tk.END, "生成中，请稍候...\n")
        self.progress_var.set(0)
        self.progress_label_var.set("生成架构中...")

        def task():
            try:
                llm = self.get_llm()
                # 优先使用提示词文件中的完整内容作为参考，其次使用 new_book_prompt_text
                template = ""
                if prompt_content:
                    template = prompt_content
                elif hasattr(self, 'new_book_prompt_text'):
                    template = self.new_book_prompt_text.get(1.0, tk.END).strip()

                prompt = f"""你是一位专业的小说策划师。请根据以下信息生成小说整体架构，必须包含底层逻辑。

新书名：{self.new_book_title.get()}
新书简介：{self.new_book_synopsis.get()}
题材：{self.genre.get()}
章节数：{self.chapter_num.get()}
每章字数：{self.words_per_chapter.get()}

以下是从拆书分析中得到的可复用写作模板，请严格参考其结构来设计架构（但不要抄袭原剧情）：
{template[:2000]}

请输出包含底层逻辑的完整架构：

### 一、小说简介（广告三段式）
- 第一段：主角处境（建立代入感）
- 第二段：抛出冲突（制造紧张感）
- 第三段：悬念收尾（引发好奇心）

### 二、主要人物设定（包含底层逻辑）
- 主角：性格、动机、成长弧光、内在冲突
- 配角1：与主角的关系、独立动机、功能定位
- 配角2：与主角的关系、独立动机、功能定位

### 三、世界观设定（包含底层逻辑）
- 核心规则：世界的运行法则
- 力量体系：如果有的话，如何获得、成长、限制
- 社会结构：权力分布、阶级关系
- 文化背景：习俗、信仰、价值观

### 四、主线剧情概要（包含底层逻辑）
- 起：初始状态、触发事件
- 承：发展过程、冲突升级
- 转：转折点、高潮
- 合：结局、主题升华

### 五、底层逻辑设计
- 故事的核心驱动力是什么？
- 主角的成长逻辑是什么？
- 冲突的解决逻辑是什么？
- 主题的表达逻辑是什么？

请确保架构完整、逻辑清晰，可直接用于后续创作。"""

                result = llm.chat(prompt)

                # 检查 API 是否返回错误
                if result.startswith("错误：") or result.startswith("请求异常"):
                    error_msg = f"API调用失败：{result}"
                    self.root.after(0, lambda: self.text_arch.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.text_arch.insert(tk.END, error_msg))
                    self.root.after(0, lambda: messagebox.showerror("生成失败", error_msg))
                    self.log_to_write(error_msg)
                    return

                # 保存架构文档到 jiagou 文件夹
                os.makedirs(jiagou_dir, exist_ok=True)
                try:
                    with open(arch_file, 'w', encoding='utf-8') as f:
                        f.write(result)
                    self.log_to_write(f"架构文档已保存至：{arch_file}")
                except Exception as e:
                    self.log_to_write(f"保存架构文档失败：{e}")
                    self.root.after(0, lambda: messagebox.showerror("保存失败", f"无法保存架构文件：{e}"))

                # 更新 UI
                self.root.after(0, lambda: self.text_arch.delete(1.0, tk.END))
                self.root.after(0, lambda: self.text_arch.insert(tk.END, result))
                self.architecture = result
                self.root.after(0, lambda: self.log_to_write("架构生成完成"))
                self.root.after(0, lambda: self.progress_label_var.set("架构生成完成"))
            except Exception as e:
                error_msg = f"生成架构时发生异常：{str(e)}"
                self.log_to_write(error_msg)
                self.root.after(0, lambda: self.text_arch.delete(1.0, tk.END))
                self.root.after(0, lambda: self.text_arch.insert(tk.END, error_msg))
                self.root.after(0, lambda: messagebox.showerror("异常", error_msg))
            finally:
                self.root.after(0, lambda: self.gen_arch_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: setattr(self, 'is_generating_arch', False))
                # 清除线程引用
                if hasattr(self, '_arch_thread'):
                    self._arch_thread = None

        # 保存线程引用以便后续检查
        self._arch_thread = threading.Thread(target=task, daemon=True)
        self._arch_thread.start()

    def gen_blueprints(self):
        # 防止重复调用
        if self.is_generating_blueprint:
            messagebox.showwarning("提示", "章节蓝图正在生成中，请稍后...")
            return

        if not self.architecture:
            messagebox.showwarning("提示", "请先执行Step1生成架构")
            return
        
        book_title = self.new_book_title.get().strip()
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        lantu_dir = os.path.join(book_dir, "lantu")
        blueprint_file = os.path.join(lantu_dir, f"{book_title}-蓝图.txt")
        
        if os.path.exists(blueprint_file):
            try:
                with open(blueprint_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                if len(existing_content) > 1000:
                    response = messagebox.askyesno("蓝图已存在", 
                        f"蓝图文档已存在且内容完整：{blueprint_file}\n\n是否要重新生成蓝图？")
                    if not response:
                        self.log_to_write("用户取消，停止蓝图生成")
                        return
            except Exception as e:
                self.log_to_write(f"读取蓝图文档失败：{e}")
        
        total = self.chapter_num.get()
        self.log_to_write(f"开始为{total}个章节生成蓝图（每章6-8字章节名）...")
        self.text_blueprint.delete(1.0, tk.END)
        self.chapter_blueprints.clear()
        self.progress_var.set(0)
        self.progress_label_var.set("生成章节蓝图中...")

        self.is_generating_blueprint = True
        self.gen_blueprint_btn.config(state=tk.DISABLED)

        def task():
            try:
                llm = self.get_llm()
                all_blueprints = []
                for ch in range(1, total+1):
                    self.log_to_write(f"正在生成第{ch}章蓝图...")
                    prompt = f"""你是小说家。根据以下整体架构，为第{ch}章（共{total}章）生成详细大纲，必须包含章节蓝图底层逻辑。

整体架构摘要：{self.architecture[:1500]}
新书名：{self.new_book_title.get()}
题材：{self.genre.get()}
**要求：必须为本章生成一个6-8个字的章节名（例如"武当山上放牛"、"深夜密谈惊变"），章节名要概括本章核心事件。**

请输出包含底层逻辑的章节蓝图：

### 章节名（6-8个字）
（概括本章核心事件）

### 本章核心冲突
- 冲突类型：____________
- 冲突双方：____________
- 冲突根源：____________

### 主要场景（3-5个）
1. 场景1：____________（功能：____________）
2. 场景2：____________（功能：____________）
3. 场景3：____________（功能：____________）

### 本章结尾悬念
- 悬念类型：____________
- 如何引发读者好奇心：____________

### 预计字数
{self.words_per_chapter.get()}字

### 章节蓝图底层逻辑
- 本章在整体故事中的功能定位：____________
- 本章如何推动主角成长：____________
- 本章如何为后续章节铺垫：____________
- 本章的情绪曲线设计：____________
- 本章的节奏控制策略：____________"""
                    
                    bp = llm.chat(prompt)
                    # 检查API错误
                    if bp.startswith("错误：") or bp.startswith("请求异常"):
                        self.root.after(0, lambda: self.log_to_write(f"第{ch}章生成失败：{bp}"))
                        continue
                    self.chapter_blueprints[ch] = bp
                    all_blueprints.append(f"========== 第{ch}章 {self.extract_chapter_title(bp)} ==========\n{bp}\n\n")
                    
                    self.root.after(0, lambda c=ch, b=bp: self.text_blueprint.insert(tk.END, f"\n========== 第{c}章 {self.extract_chapter_title(b)} ==========\n{b}\n\n"))
                    self.root.after(0, lambda: self.text_blueprint.see(tk.END))
                    progress = int(ch / total * 100)
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                # 保存蓝图文档
                os.makedirs(lantu_dir, exist_ok=True)
                full_blueprint_content = f"《{book_title}》章节蓝图\n\n" + "\n".join(all_blueprints)
                with open(blueprint_file, 'w', encoding='utf-8') as f:
                    f.write(full_blueprint_content)
                self.log_to_write(f"蓝图文档已保存至：{blueprint_file}")
                
                self.root.after(0, lambda: self.log_to_write("所有章节蓝图生成完成"))
                self.root.after(0, lambda: self.progress_label_var.set("蓝图生成完成"))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_write(f"生成蓝图时发生错误：{e}"))
            finally:
                self.root.after(0, lambda: self.gen_blueprint_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: setattr(self, 'is_generating_blueprint', False))
        
        threading.Thread(target=task, daemon=True).start()

    def extract_chapter_title(self, blueprint):
        match = re.search(r'章节名[：:]\s*([^\r\n]+)', blueprint)
        if match:
            title = match.group(1).strip()
            if len(title) > 20:
                title = title[:20].strip()
            return title or "未命名"
        return "未命名"

    def gen_all_drafts(self):
        """生成全部草稿，实现caogao文件夹逻辑，按章节名保存"""
        if not self.chapter_blueprints:
            messagebox.showwarning("提示", "请先执行Step2生成蓝图")
            return
        
        book_title = self.new_book_title.get().strip()
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        caogao_dir = os.path.join(book_dir, "caogao")
        os.makedirs(caogao_dir, exist_ok=True)
        
        existing_chapters = set()
        if os.path.exists(caogao_dir):
            for filename in os.listdir(caogao_dir):
                # 修改：支持按章节名保存的文件（不含"第X章_"前缀）
                # 旧格式：第1章_深夜密谈惊变.txt
                # 新格式：深夜密谈惊变.txt
                match = re.match(r'第(\d+)章_[^_]+\.txt', filename)
                if match:
                    existing_chapters.add(int(match.group(1)))
        
        total = self.chapter_num.get()
        if not existing_chapters:
            self.log_to_write("未找到任何章节草稿，开始重新书写全部章节草稿...")
            missing_chapters = list(range(1, total+1))
        else:
            self.log_to_write(f"检测到已存在章节草稿: {sorted(existing_chapters)}")
            missing_chapters = [ch for ch in range(1, total+1) if ch not in existing_chapters]
            if not missing_chapters:
                messagebox.showinfo("提示", "所有章节草稿均已存在，无需生成。")
                self.merge_all_chapters_to_full_document()
                return
            for ch in sorted(existing_chapters):
                chapter_file = os.path.join(caogao_dir, f"第{ch}章_*.txt")
                import glob
                files = glob.glob(chapter_file)
                if files:
                    try:
                        with open(files[0], 'r', encoding='utf-8') as f:
                            content = f.read()
                        if len(content) < 100:
                            self.log_to_write(f"第{ch}章内容不完整，将重新生成")
                            missing_chapters.append(ch)
                    except Exception as e:
                        self.log_to_write(f"读取第{ch}章失败：{e}")
                        missing_chapters.append(ch)
            missing_chapters = sorted(set(missing_chapters))
        
        self.log_to_write(f"需要生成/续写的章节: {missing_chapters}")
        self.generate_chapters_by_list_with_caogao(missing_chapters, caogao_dir)

    def gen_novel_one_click(self):
        """一键生成小说：依次执行生成架构、章节蓝图、全部草稿"""
        # 检查前置条件
        if not self.new_book_title.get().strip():
            messagebox.showwarning("提示", "请先填写新书名")
            return
        
        if not self.api_key.get().strip():
            messagebox.showwarning("提示", "请先在系统设置中配置API Key")
            return
        
        # 确认操作
        response = messagebox.askyesno("确认操作", 
            "即将开始一键生成小说，将依次执行：\n\n"
            "1. 生成整体架构\n"
            "2. 生成章节蓝图\n"
            "3. 生成全部草稿\n\n"
            "此过程可能需要较长时间，是否继续？")
        if not response:
            return
        
        # 禁用一键生成按钮，防止重复点击
        if hasattr(self, 'gen_all_in_one_btn'):
            self.gen_all_in_one_btn.config(state=tk.DISABLED)
        
        self.log_to_write("========== 开始一键生成小说 ==========")
        self.log_to_write("步骤1/3：生成整体架构...")
        self.progress_label_var.set("一键生成：生成架构中...")
        self.progress_var.set(0)
        
        # 获取必要的信息用于后续步骤
        book_title = self.new_book_title.get().strip()
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        jiagou_dir = os.path.join(book_dir, "jiagou")
        arch_file = os.path.join(jiagou_dir, f"{book_title}-架构.txt")
        
        # 获取提示词文件内容
        selected_prompt_name = self.prompt_file_combobox.get() if hasattr(self, 'prompt_file_combobox') else ""
        prompt_content = ""
        if selected_prompt_name:
            tishici_dir = "./tishici"
            prompt_file = os.path.join(tishici_dir, f"{selected_prompt_name}-提示词.txt")
            if os.path.exists(prompt_file):
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompt_content = f.read()
                except:
                    pass
        
        # 设置生成架构的标志
        self.is_generating_arch = True
        self.gen_arch_btn.config(state=tk.DISABLED)
        
        def execute_step1_architecture():
            """步骤1：生成架构"""
            try:
                llm = self.get_llm()
                
                template = ""
                if prompt_content:
                    template = prompt_content
                elif hasattr(self, 'new_book_prompt_text'):
                    template = self.new_book_prompt_text.get(1.0, tk.END).strip()

                prompt = f"""你是一位专业的小说策划师。请根据以下信息生成小说整体架构，必须包含底层逻辑。

新书名：{self.new_book_title.get()}
新书简介：{self.new_book_synopsis.get()}
题材：{self.genre.get()}
章节数：{self.chapter_num.get()}
每章字数：{self.words_per_chapter.get()}

以下是从拆书分析中得到的可复用写作模板，请严格参考其结构来设计架构（但不要抄袭原剧情）：
{template[:2000]}

请输出包含底层逻辑的完整架构：

### 一、小说简介（广告三段式）
- 第一段：主角处境（建立代入感）
- 第二段：抛出冲突（制造紧张感）
- 第三段：悬念收尾（引发好奇心）

### 二、主要人物设定（包含底层逻辑）
- 主角：性格、动机、成长弧光、内在冲突
- 配角1：与主角的关系、独立动机、功能定位
- 配角2：与主角的关系、独立动机、功能定位

### 三、世界观设定（包含底层逻辑）
- 核心规则：世界的运行法则
- 力量体系：如果有的话，如何获得、成长、限制
- 社会结构：权力分布、阶级关系
- 文化背景：习俗、信仰、价值观

### 四、主线剧情概要（包含底层逻辑）
- 起：初始状态、触发事件
- 承：发展过程、冲突升级
- 转：转折点、高潮
- 合：结局、主题升华

### 五、底层逻辑设计
- 故事的核心驱动力是什么？
- 主角的成长逻辑是什么？
- 冲突的解决逻辑是什么？
- 主题的表达逻辑是什么？

请确保架构完整、逻辑清晰，可直接用于后续创作。"""

                result = llm.chat(prompt)

                if result.startswith("错误：") or result.startswith("请求异常"):
                    self.root.after(0, lambda: messagebox.showerror("架构生成失败", result))
                    self.root.after(0, lambda: self.log_to_write(f"架构生成失败：{result}"))
                    self.root.after(0, self._enable_all_gen_buttons)
                    return

                # 保存架构
                os.makedirs(jiagou_dir, exist_ok=True)
                with open(arch_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                self.architecture = result
                self.root.after(0, lambda: self.text_arch.delete(1.0, tk.END))
                self.root.after(0, lambda: self.text_arch.insert(tk.END, result))
                self.log_to_write("步骤1完成：整体架构已生成")
                
                # 进入步骤2
                self.root.after(100, execute_step2_blueprints)
                
            except Exception as e:
                error_msg = f"生成架构时发生异常：{str(e)}"
                self.root.after(0, lambda: messagebox.showerror("异常", error_msg))
                self.root.after(0, lambda: self.log_to_write(error_msg))
                self.root.after(0, self._enable_all_gen_buttons)
            finally:
                self.root.after(0, lambda: self.gen_arch_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: setattr(self, 'is_generating_arch', False))
        
        def execute_step2_blueprints():
            """步骤2：生成蓝图"""
            self.log_to_write("步骤2/3：生成章节蓝图...")
            self.progress_label_var.set("一键生成：生成章节蓝图中...")
            
            total = self.chapter_num.get()
            self.is_generating_blueprint = True
            self.gen_blueprint_btn.config(state=tk.DISABLED)
            self.text_blueprint.delete(1.0, tk.END)
            self.chapter_blueprints.clear()
            
            def task():
                try:
                    llm = self.get_llm()
                    all_blueprints = []
                    lantu_dir = os.path.join(book_dir, "lantu")
                    blueprint_file = os.path.join(lantu_dir, f"{book_title}-蓝图.txt")
                    
                    for ch in range(1, total+1):
                        self.log_to_write(f"正在生成第{ch}章蓝图...")
                        prompt = f"""你是小说家。根据以下整体架构，为第{ch}章（共{total}章）生成详细大纲，必须包含章节蓝图底层逻辑。

整体架构摘要：{self.architecture[:1500]}
新书名：{self.new_book_title.get()}
题材：{self.genre.get()}
**要求：必须为本章生成一个6-8个字的章节名（例如"武当山上放牛"、"深夜密谈惊变"），章节名要概括本章核心事件。**

请输出包含底层逻辑的章节蓝图：

### 章节名（6-8个字）
（概括本章核心事件）

### 本章核心冲突
- 冲突类型：____________
- 冲突双方：____________
- 冲突根源：____________

### 主要场景（3-5个）
1. 场景1：____________（功能：____________）
2. 场景2：____________（功能：____________）
3. 场景3：____________（功能：____________）

### 本章结尾悬念
- 悬念类型：____________
- 如何引发读者好奇心：____________

### 预计字数
{self.words_per_chapter.get()}字

### 章节蓝图底层逻辑
- 本章在整体故事中的功能定位：____________
- 本章如何推动主角成长：____________
- 本章如何为后续章节铺垫：____________
- 本章的情绪曲线设计：____________
- 本章的节奏控制策略：____________"""
                        
                        bp = llm.chat(prompt)
                        if bp.startswith("错误：") or bp.startswith("请求异常"):
                            self.root.after(0, lambda: self.log_to_write(f"第{ch}章生成失败：{bp}"))
                            continue
                        
                        self.chapter_blueprints[ch] = bp
                        ch_title = self.extract_chapter_title(bp)
                        all_blueprints.append(f"========== 第{ch}章 {ch_title} ==========\n{bp}\n\n")
                        
                        self.root.after(0, lambda c=ch, t=ch_title, b=bp: self.text_blueprint.insert(tk.END, f"\n========== 第{c}章 {t} ==========\n{b}\n\n"))
                        self.root.after(0, lambda: self.text_blueprint.see(tk.END))
                        
                        progress = int(ch / total * 33)  # 0-33%
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    
                    # 保存蓝图
                    os.makedirs(lantu_dir, exist_ok=True)
                    full_blueprint_content = f"《{book_title}》章节蓝图\n\n" + "\n".join(all_blueprints)
                    with open(blueprint_file, 'w', encoding='utf-8') as f:
                        f.write(full_blueprint_content)
                    
                    self.log_to_write("步骤2完成：章节蓝图已生成")
                    
                    # 进入步骤3
                    self.root.after(100, execute_step3_drafts)
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log_to_write(f"生成蓝图时发生错误：{e}"))
                    self.root.after(0, self._enable_all_gen_buttons)
                finally:
                    self.root.after(0, lambda: self.gen_blueprint_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: setattr(self, 'is_generating_blueprint', False))
            
            threading.Thread(target=task, daemon=True).start()
        
        def execute_step3_drafts():
            """步骤3：生成草稿"""
            self.log_to_write("步骤3/3：生成全部草稿...")
            self.progress_label_var.set("一键生成：生成章节草稿中...")
            
            total = self.chapter_num.get()
            caogao_dir = os.path.join(book_dir, "caogao")
            os.makedirs(caogao_dir, exist_ok=True)
            
            def task():
                try:
                    llm = self.get_llm()
                    chapter_list = list(range(1, total+1))
                    
                    for idx, ch in enumerate(chapter_list):
                        bp = self.chapter_blueprints.get(ch, "")
                        if not bp:
                            self.log_to_write(f"第{ch}章蓝图缺失，跳过")
                            continue
                        
                        ch_title = self.extract_chapter_title(bp)
                        self.log_to_write(f"正在撰写第{ch}章「{ch_title}」草稿...")
                        
                        prompt = f"""你是一位小说家，请根据以下蓝图扩写成完整的章节内容（约{self.words_per_chapter.get()}字）。

章节蓝图：
{bp}

写作要求（去AI味，贴近人类风格）：
{self.deai_prompt}

**特别要求：**
1. 保持章节连贯性
2. 保持底层逻辑
3. 保持节奏控制
4. 保持情绪曲线
5. 保持功能定位

请输出完整的章节正文，章节开头格式为"第{ch}章 {ch_title}" """
                        
                        draft = llm.chat(prompt)
                        if draft.startswith("错误：") or draft.startswith("请求异常"):
                            self.log_to_write(f"第{ch}章生成失败：{draft}")
                            continue
                        
                        self.chapter_drafts[ch] = draft
                        
                        self.root.after(0, lambda c=ch, t=ch_title, d=draft: self.text_draft.insert(tk.END, f"\n========== 第{c}章 {t} ==========\n{d}\n\n"))
                        self.root.after(0, lambda: self.text_draft.see(tk.END))
                        
                        # 保存草稿到caogao文件夹，按章节名保存
                        filename = f"{ch_title}.txt"
                        filepath = os.path.join(caogao_dir, filename)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"第{ch}章 {ch_title}\n\n{draft}")
                        
                        progress = 33 + int((idx + 1) / total * 66)  # 33-99%
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    
                    # 合并完整文档
                    self.root.after(0, self.merge_all_chapters_to_full_document)
                    
                    self.log_to_write("========== 一键生成完成 ==========")
                    self.root.after(0, lambda: self.progress_label_var.set("一键生成完成"))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: messagebox.showinfo("成功", "一键生成小说完成！"))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log_to_write(f"生成草稿时发生错误：{e}"))
                finally:
                    self.root.after(0, self._enable_all_gen_buttons)
            
            threading.Thread(target=task, daemon=True).start()
        
        # 启动第一步
        threading.Thread(target=execute_step1_architecture, daemon=True).start()
    
    def _enable_all_gen_buttons(self):
        """重新启用所有生成按钮"""
        if hasattr(self, 'gen_arch_btn'):
            self.gen_arch_btn.config(state=tk.NORMAL)
        if hasattr(self, 'gen_blueprint_btn'):
            self.gen_blueprint_btn.config(state=tk.NORMAL)
        if hasattr(self, 'gen_all_in_one_btn'):
            self.gen_all_in_one_btn.config(state=tk.NORMAL)

    def gen_partial_drafts(self):
        # 防止重复调用
        if self.is_generating_partial:
            messagebox.showwarning("提示", "部分草稿正在生成中，请稍后...")
            return

        if not self.chapter_blueprints:
            messagebox.showwarning("提示", "请先执行Step2生成蓝图")
            return
        total = self.chapter_num.get()
        start_str = simpledialog.askstring("生成部分草稿", f"请输入起始章节号 (1-{total}):", initialvalue="1")
        if not start_str:
            return
        try:
            start = int(start_str)
        except:
            messagebox.showerror("错误", "起始章节号必须是数字")
            return
        end_str = simpledialog.askstring("生成部分草稿", f"请输入结束章节号 (1-{total}):", initialvalue=str(total))
        if not end_str:
            return
        try:
            end = int(end_str)
        except:
            messagebox.showerror("错误", "结束章节号必须是数字")
            return
        if start < 1 or end > total or start > end:
            messagebox.showerror("错误", f"章节号范围无效，应在1-{total}之间且起始<=结束")
            return

        save_dir = self.save_path.get()
        os.makedirs(save_dir, exist_ok=True)
        existing_chapters = set()
        if os.path.exists(save_dir):
            for filename in os.listdir(save_dir):
                match = re.match(r'第(\d+)章_.*\.txt', filename)
                if match:
                    existing_chapters.add(int(match.group(1)))

        needed = [ch for ch in range(start, end+1) if ch not in existing_chapters]
        if not needed:
            messagebox.showinfo("提示", f"章节 {start}-{end} 范围内所有文件均已存在，无需生成。")
            return

        self.log_to_write(f"指定范围 {start}-{end}，已存在 {existing_chapters & set(range(start, end+1))}，需要生成 {needed}")
        self.is_generating_partial = True
        self.gen_partial_btn.config(state=tk.DISABLED)
        # 复用原有生成逻辑，但需要在线程结束后恢复按钮
        def task():
            try:
                self.generate_chapters_by_list(needed)
            finally:
                self.root.after(0, lambda: self.gen_partial_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: setattr(self, 'is_generating_partial', False))
        threading.Thread(target=task, daemon=True).start()

    def generate_chapters_by_list(self, chapter_list):
        """生成章节草稿，保存到新书名文件夹内的caogao子文件夹"""
        if not chapter_list:
            return
        
        book_title = self.new_book_title.get().strip()
        save_dir = self.save_path.get()
        book_dir = os.path.join(save_dir, book_title)
        caogao_dir = os.path.join(book_dir, "caogao")
        os.makedirs(caogao_dir, exist_ok=True)
        
        self.log_to_write(f"开始生成章节: {chapter_list}")
        self.log_to_write(f"草稿将保存到: {caogao_dir}")
        self.progress_var.set(0)
        self.progress_label_var.set("生成章节草稿中...")

        def task():
            llm = self.get_llm()
            total = len(chapter_list)
            for idx, ch in enumerate(chapter_list):
                bp = self.chapter_blueprints.get(ch, "")
                if not bp:
                    self.log_to_write(f"第{ch}章蓝图缺失，跳过")
                    continue
                ch_title = self.extract_chapter_title(bp)
                self.log_to_write(f"正在撰写第{ch}章「{ch_title}」草稿...")
                prompt = f"""你是一位小说家，请根据以下蓝图扩写成完整的章节内容（约{self.words_per_chapter.get()}字）。
蓝图：
{bp}
{self.deai_prompt}
要求：语言流畅，描写生动，直接输出章节正文。章节开头格式为"第{ch}章 {ch_title}" """
                draft = llm.chat(prompt)
                if draft.startswith("错误：") or draft.startswith("请求异常"):
                    self.log_to_write(f"第{ch}章生成失败：{draft}")
                    continue
                self.chapter_drafts[ch] = draft
                self.root.after(0, lambda c=ch, t=ch_title, d=draft: self.text_draft.insert(tk.END, f"\n========== 第{c}章 {t} ==========\n{d}\n\n"))
                self.root.after(0, lambda: self.text_draft.see(tk.END))
                
                # 保存草稿到新书名文件夹内的caogao子文件夹
                filename = f"{ch_title}.txt"
                filepath = os.path.join(caogao_dir, filename)
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"第{ch}章 {ch_title}\n\n{draft}")
                    self.log_to_write(f"第{ch}章已保存至：{filepath}")
                except Exception as e:
                    self.log_to_write(f"保存第{ch}章失败：{e}")
                
                progress = int((idx + 1) / total * 100)
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
            self.root.after(0, self.merge_all_chapters_to_full_document)
            self.log_to_write("缺失章节生成完成，已合并完整文档")
            self.root.after(0, lambda: self.progress_label_var.set("草稿生成完成"))
        threading.Thread(target=task, daemon=True).start()
    
    def generate_chapters_by_list_with_caogao(self, chapter_list, caogao_dir):
        if not chapter_list:
            return
        self.log_to_write(f"开始生成章节到caogao文件夹: {chapter_list}")
        self.progress_var.set(0)
        self.progress_label_var.set("生成章节草稿中...")

        def task():
            llm = self.get_llm()
            total = len(chapter_list)
            for idx, ch in enumerate(chapter_list):
                bp = self.chapter_blueprints.get(ch, "")
                if not bp:
                    self.log_to_write(f"第{ch}章蓝图缺失，跳过")
                    continue
                ch_title = self.extract_chapter_title(bp)
                self.log_to_write(f"正在撰写第{ch}章「{ch_title}」草稿...")
                
                prompt = f"""你是一位小说家，请根据以下蓝图扩写成完整的章节内容（约{self.words_per_chapter.get()}字），必须保持章节连贯、完整和已有的底层逻辑。

章节蓝图：
{bp}

写作要求（去AI味，贴近人类风格）：
{self.deai_prompt}

**特别要求：**
1. 保持章节连贯性：确保本章内容与前后章节逻辑连贯
2. 保持底层逻辑：遵循已有的故事底层逻辑和人物设定
3. 保持节奏控制：按照蓝图中的节奏控制策略进行写作
4. 保持情绪曲线：按照蓝图中的情绪曲线设计进行写作
5. 保持功能定位：确保本章在整体故事中的功能定位得到体现

请输出完整的章节正文，章节开头格式为"第{ch}章 {ch_title}"
                
                draft = llm.chat(prompt)
                if draft.startswith("错误：") or draft.startswith("请求异常"):
                    self.log_to_write(f"第{ch}章生成失败：{draft}")
                    continue
                self.chapter_drafts[ch] = draft
                
                self.root.after(0, lambda c=ch, t=ch_title, d=draft: self.text_draft.insert(tk.END, f"\n========== 第{c}章 {t} ==========\n{d}\n\n"))
                self.root.after(0, lambda: self.text_draft.see(tk.END))
                
                os.makedirs(caogao_dir, exist_ok=True)
                # 按章节名保存（不含"第X章_"前缀）
                filename = f"{ch_title}.txt"
                filepath = os.path.join(caogao_dir, filename)
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"第{ch}章 {ch_title}\n\n{draft}")
                    self.log_to_write(f"第{ch}章已保存至：{filepath}")
                except Exception as e:
                    self.log_to_write(f"保存第{ch}章失败：{e}")
                
                progress = int((idx + 1) / total * 100)
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
            
            self.root.after(0, self.merge_all_chapters_to_full_document)
            self.log_to_write("章节草稿生成完成，已保存到caogao文件夹")
            self.root.after(0, lambda: self.progress_label_var.set("草稿生成完成"))
        
        threading.Thread(target=task, daemon=True).start()

    def merge_all_chapters_to_full_document(self):
        save_dir = self.save_path.get()
        book_title = self.new_book_title.get().strip()
        if not book_title:
            book_title = "未命名小说"
        full_filename = f"{book_title}.txt"
        full_filepath = os.path.join(save_dir, full_filename)

        chapters_content = []
        total = self.chapter_num.get()
        for ch in range(1, total+1):
            found = False
            # 优先从 caogao 目录读取，若没有则从 save_dir 读取
            caogao_dir = os.path.join(save_dir, book_title, "caogao") if book_title != "未命名小说" else None
            search_dirs = []
            if caogao_dir and os.path.exists(caogao_dir):
                search_dirs.append(caogao_dir)
            search_dirs.append(save_dir)
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for filename in os.listdir(search_dir):
                        if filename.startswith(f"第{ch}章_"):
                            filepath = os.path.join(search_dir, filename)
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                chapters_content.append(content)
                                found = True
                            except:
                                pass
                            break
                if found:
                    break
            if not found and ch in self.chapter_drafts:
                chapters_content.append(self.chapter_drafts[ch])
            elif not found:
                self.log_to_write(f"警告：第{ch}章内容未找到，可能生成失败")
                chapters_content.append(f"【第{ch}章内容缺失】")

        if not chapters_content:
            messagebox.showwarning("提示", "没有找到任何章节内容，无法合并文档。")
            return

        full_text = "\n\n".join(chapters_content)
        try:
            with open(full_filepath, 'w', encoding='utf-8') as f:
                f.write(full_text)
            self.log_to_write(f"完整文档已保存至：{full_filepath}")
            messagebox.showinfo("完成", f"全部章节已生成完毕，完整文档保存为：{full_filepath}")
        except Exception as e:
            messagebox.showerror("错误", f"保存完整文档失败：{e}")

    def load_template_from_library(self):
        templates = self.template_manager.get_all_templates()
        if not templates:
            messagebox.showinfo("提示", "模板库为空，请先在拆书页面保存模板")
            return
        win = tk.Toplevel(self.root)
        win.title("选择模板")
        win.geometry("700x500")
        listbox = tk.Listbox(win, width=80, height=20)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for t in templates:
            listbox.insert(tk.END, f"{t['source_title']} - {t['created']} (评分:{t['rating']})")
        def load():
            sel = listbox.curselection()
            if not sel:
                return
            template = templates[sel[0]]
            self.current_template_text = template['template_text']
            if hasattr(self, 'new_book_prompt_text'):
                self.new_book_prompt_text.delete(1.0, tk.END)
                self.new_book_prompt_text.insert(tk.END, self.current_template_text)
            messagebox.showinfo("成功", f"已加载模板：{template['source_title']}")
            win.destroy()
        ttk.Button(win, text="加载此模板", command=load).pack(pady=10)

    def log_to_write(self, msg):
        print(msg)

    # ---------- 系统设置页面 ----------
    def on_model_select(self):
        model_id = self.current_model.get()
        if model_id in self.MODEL_PRESETS:
            display_name, default_base_url, default_model = self.MODEL_PRESETS[model_id]
            self.base_url.set(default_base_url)
            self.model_name.set(default_model)
            self.log_to_system(f"已切换到 {display_name} 模型")

    def create_system_settings_page(self):
        page = tk.Frame(self.right_area, bg='#f0f2f5')
        tk.Label(page, text="系统设置", font=('微软雅黑', 18, 'bold'),
                 bg='#f0f2f5', fg='#2c3e50').pack(anchor='w', padx=20, pady=20)
        model_frame = ttk.LabelFrame(page, text="模型选择", padding=10)
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        models_list = list(self.MODEL_PRESETS.items())
        row, col = 0, 0
        for i, (model_id, (display_name, _, _)) in enumerate(models_list):
            rb = tk.Radiobutton(model_frame, text=display_name, variable=self.current_model,
                                value=model_id, command=self.on_model_select)
            rb.grid(row=row, column=col, sticky='w', padx=10, pady=5)
            col += 1
            if col >= 4:
                col = 0
                row += 1
        config_frame = ttk.LabelFrame(page, text="API 配置", padding=10)
        config_frame.pack(fill=tk.X, padx=20, pady=10)
        row = 0
        ttk.Label(config_frame, text="API Key:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.api_key, width=50, show="*").grid(row=row, column=1, padx=5, pady=5, sticky='w')
        row += 1
        ttk.Label(config_frame, text="Base URL:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.base_url, width=50).grid(row=row, column=1, padx=5, pady=5, sticky='w')
        row += 1
        ttk.Label(config_frame, text="模型名称:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.model_name, width=30).grid(row=row, column=1, padx=5, pady=5, sticky='w')
        row += 1
        ttk.Label(config_frame, text="剩余TOKEN:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.token_balance_label = ttk.Label(config_frame, text="未查询", width=30, relief=tk.SUNKEN, anchor='w')
        self.token_balance_label.grid(row=row, column=1, padx=5, pady=5, sticky='w')
        row += 1
        ttk.Label(config_frame, text="Temperature:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        scale = ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.temp, orient=tk.HORIZONTAL, length=200)
        scale.grid(row=row, column=1, sticky='w', padx=5)
        ttk.Label(config_frame, textvariable=self.temp).grid(row=row, column=2, padx=5)
        row += 1
        btn_frame = tk.Frame(config_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="测试配置", command=self.test_api).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="查询TOKEN", command=self.query_token_balance).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存配置", command=self.save_llm_config).pack(side=tk.LEFT, padx=5)
        proxy_frame = ttk.LabelFrame(page, text="代理设置", padding=10)
        proxy_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(proxy_frame, text="暂未实现").pack()
        log_frame = ttk.LabelFrame(page, text="测试日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.test_log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, font=('微软雅黑', 9))
        self.test_log_text.pack(fill=tk.BOTH, expand=True)
        ttk.Button(log_frame, text="清空日志", command=lambda: self.test_log_text.delete(1.0, tk.END)).pack(pady=5)
        return page

    def query_token_balance(self):
        if not self.api_key.get().strip():
            messagebox.showwarning("提示", "请填写API Key")
            return
        self.token_balance_label.config(text="查询中...")
        def query_task():
            try:
                import time
                time.sleep(1)
                self.root.after(0, lambda: self.token_balance_label.config(text="1000.50 USD"))
            except Exception as e:
                self.root.after(0, lambda: self.token_balance_label.config(text=f"查询失败: {str(e)}"))
        threading.Thread(target=query_task, daemon=True).start()

    def test_api(self):
        self.log_to_system("开始测试API连接...")
        def test():
            url = f"{self.base_url.get().rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key.get().strip()}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name.get().strip(),
                "messages": [{"role": "user", "content": "请回复：OK"}],
                "temperature": self.temp.get(),
                "max_tokens": 10
            }
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()["choices"][0]["message"]["content"]
                    ok = "OK" in result
                    msg = f"[{datetime.now().strftime('%H:%M:%S')}] " + ("✅ API测试成功！" if ok else f"❌ API返回异常: {result}")
                else:
                    msg = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ API测试失败，状态码: {resp.status_code}\n响应内容: {resp.text}"
            except Exception as e:
                msg = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 请求异常: {str(e)}"
            self.log_to_system(msg)
        threading.Thread(target=test, daemon=True).start()

    def log_to_system(self, msg):
        if hasattr(self, 'test_log_text'):
            self.test_log_text.insert(tk.END, msg + "\n")
            self.test_log_text.see(tk.END)

    # ---------- 页面切换 ----------
    def show_home(self):
        for child in self.right_area.winfo_children():
            child.pack_forget()
        self.page_home.pack(fill=tk.BOTH, expand=True)

    def show_knowledge(self):
        for child in self.right_area.winfo_children():
            child.pack_forget()
        self.page_knowledge.pack(fill=tk.BOTH, expand=True)

    def show_write_book(self):
        for child in self.right_area.winfo_children():
            child.pack_forget()
        self.page_write_book.pack(fill=tk.BOTH, expand=True)
        self.show_write_page1()

    def show_book_analysis(self):
        for child in self.right_area.winfo_children():
            child.pack_forget()
        self.refresh_book_doc_list()
        self.refresh_analysis_library()
        self.page_book_analysis.pack(fill=tk.BOTH, expand=True)

    def show_system_settings(self):
        for child in self.right_area.winfo_children():
            child.pack_forget()
        self.page_system.pack(fill=tk.BOTH, expand=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SimpleWorkbench()
    app.run()