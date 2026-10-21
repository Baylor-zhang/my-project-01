import os.path
import json
from datetime import datetime
from typing import Any
from openai import OpenAI
from pydantic import BaseModel
from fastapi import FastAPI,Request
# 新增下面这一行
from fastapi.responses import FileResponse
from rich.json import JSON
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging


#日志记录
#为了能够灵活的控制项目中日志的输出，我们可以通过官方提供的loging模块 来输出日志，具体做法如下：
#日志级别：DEBUG < INFO < WARNING < ERROR < FATAL
#DEBUG：调试阶段，记录所有信息
#INFO：记录常规的告警信息
#WARNING：记录警告信息，但是不记录DEBUG和INFO级别信息
#ERROR：记录错误信息，但是不记录DEBUG、INFO和WARNING级别信息
#FATAL：记录致命错误信息，但是不记录DEBUG、INFO、WARNING和ERROR级别信息
#配置日志的基本信息
#%(asctime)s 指的是时间
#%(levelname)s 输出日志级别
#%(filename)s 输出文件名
#%(lineno)d 获取行号
#%(message)s 输出日志信息
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d -%(message)s')

#系统提示词
# 系统提示词 - 适配DeepSeek V4
SYSTEM_PROMPT = """
# 角色定义
你是一个专门玩猜字谜的AI小助手，只进行字谜互动，不闲聊无关内容，全程纯文本交互，不使用表情符号。

## 核心能力
- 出字谜、判对错、给提示
- 记忆已用谜题，确保会话内不重复
- 简洁明快回应

## 出题规则（严格执行！）
1. 开场先友好打招呼，并随机出一道常见、简单、适合大众并必须符合逻辑推理的字谜，禁止使用生僻、低俗、网络烂梗。
2. 题目格式：“谜面”（打一字）。
3. 每次出题必须完全随机，禁止重复使用相同题目，也可以偶尔穿插使用，下面示例中的谜语。
4. 新出题目时, 不要提示, 用户需要提示时, 或者答错时, 再给予合理的提示。

## 判题规则（严格执行！）
1. 用户只回复一个字时，直接视为答案。
2. 答对：立即夸奖并揭晓谜底，格式如“太棒了！就是‘X’字！要不要再来一题？”
3. 答错：告知不对，可给一句简短提示，但不泄露答案。格式如“不对哦，再想想~”
4. 严禁在用户答错后直接公布答案！只有用户说“公布答案”或“不知道”等情况时才公布。

## 互动流程
1. 用户答对：夸奖 + 确认正确 + 询问“要不要再来一题？”
2. 用户答错：告知不对 + 简单提示 + 鼓励继续猜
3. 用户说“提示一下”：给出简短线索，不公布答案
4. 用户说“公布答案”或“不知道”：揭晓谜底并解释 + 询问“要不要再来一题？”
5. 用户说“换一题”“再来一题”：立即更换新字谜

## 回复风格约束
- 语气轻松有趣，但保持简洁
- 全程只围绕字谜，拒绝回答其他问题
- 回复不超过3句话
- **绝对不要在回复中说“这个出过了，我来个新的”或类似表述** — 直接给出新谜语即可
- 判题错误零容忍，不确定谜底时，先回复“我再想想”而不是乱判

## 常见谜语类型及谜底参考示例, 仅仅为参照示例
### 组合类
- 「一加一不是二」= 王
- 「二人不是天」= 夫
- 「十口不是田」= 古

### 包含类
- 「一人在内」= 肉
- 「口里有人」= 囚
- 「门里有口」= 问
- 「田里长草」= 苗
- 「心里有你」= 您
- 「山里有山」= 出
- 「王头上有人」= 全
- 「水上有石」= 泵

### 半取类
- 「半吃半拿」= 哈
- 「半真半假」= 值
- 「半青半紫」= 素
- 「半朋半友」= 有
- 「半推半就」= 扰
- 「半山半水」= 汕

### 象形类
- 「三人又重逢」= 众
- 「一口咬掉牛尾巴」= 告
- 「两座山」= 出
- 「三日又重逢」= 晶
"""
#创建与AI大模型交互的客户端对象(Deepseek_api_key  环境变量名字，值就是在deepseek官网中生成的API_key，如果这里不想添加具体的API_key可以将API_KEY写在环境本地电脑的环境变量中)
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'),base_url="https://api.deepseek.com")


#创建fastapi实例
app = FastAPI(title="FastAPI案例1")

#挂载静态资源文件目录.app.mount指的是在app.mount下面挂在一个目录，/static指的是任何访问/static路径的请求都到这里来处理,resource/static指的是请求的静态资源文件目录,name是FastAPI内部使用的别名
app.mount("/static",StaticFiles(directory="resource/static"),name="static")
#BaseModel类是FastAPI中用于定义数据模型的类，它继承了BaseModel类，BaseModel类是Pydantic库中用于定义数据模型的类,用于定义FastAPI中的数据类型
#通过查看浏览器的请求头信息，我们知道，有一个请求头是post类型，在RESTful API中，post请求一般用于创建资源，请求路径是/api/sessions
#所以我们开发新建会话功能的时候首先顶一个新建会话的API的POST接口，且路径为/api/sessions,然后再创建一个新建会话的接口函数
#创建会话存放的目录,检查目录是否存在，如果不存在，再创建目录
if not os.path.exists("sessions"):
    os.mkdir("sessions")

#生成会话标识
def generate_session_id():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")
#根据session_id获取到会话文件的文件名
def get_session_file_name(session_id):
    return f"sessions/{session_id}.json"

#定义数据模型
class APIresponse(BaseModel):
    code:int
    message:str
    data:Any        #表示任意类型的数据

##session_id和message为请求体中的参数内容,定义数据模型，用于接收session_id和message参数
class ChatRequest(BaseModel):
    session_id:str
    message:str

#定义路径操作参数  -->http://127.0.0.1:8000/
@app.get("/")
def root():
    #FileResponse函数的作用是返回静态文件
    logging.info("访问了根路径,返回文件")
    return FileResponse("resource/static/index.html")


#创建会话的函数
@app.post("/api/sessions")
def create_session():
    logging.info("创建会话")

    #1.生成会话的标识(会话的ID)
    session_id = generate_session_id()

    #2.组装会话的信息，保存到文件中
    session_data = {"current_session":session_id,"messages":[]}
    with open(f"sessions/{session_id}.json","w",encoding="utf-8") as f:
        #json.dump()函数的作用是把Python对象转换成JSON对象,indent参数是缩进，ensure_ascii参数是保证中文不乱码
        json.dump(session_data,f,ensure_ascii=False,indent=2)
    #3.返回数据
    # return {"code":200,"message":"会话创建成功","data":session_id}
    #3.返回值，基于对象的属性创建返回数据，但是需要提前定义数据模型Model，这里返回给客户端的数据为对象，但是BaseModel会调用Pydantic库，将数据模型转换成字典，然后返回给客户端
    return APIresponse(code=200,message="会话创建成功",data=session_id)

#AI会话交互功能
@app.post("/api/chat")
def chat(request:ChatRequest): #将它们作为参数传递给FastAPI发给送服务器端,定义request参数的值为ChatRequest对象,格式为JSON格式，这里接收到的信息为用户输入的信息
    logging.info(f"开始对话，与AI交互:{request.session_id}:{request.message}")     #在pycharm中输出信息
    # return  APIresponse(code=200,message="AI会话请求相应成功",data="AI大模型返回的数据")  #从服务器段返回消息给客户端，可以在浏览器的resopnse中查看

    #与AI大模型交互的逻辑实现
    #1.加载json文件中的会话数据
    #获取会话文件存放路径
    session_path = get_session_file_name(request.session_id)
    with open(session_path,"r",encoding="utf-8") as f:
        session_data = json.load(f)

    #2.构建AI大模型交互的会话数据
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    #追加历史会话消息到新的会话文件中
    for message in session_data["messages"]:
        messages.append(message)
    #将本次循环user输入的消息发送给大模型
    messages.append({"role":"user","content":request.message})

    #3.调用deepseek API
    # 与AI大模型进行交互的参数信息,temperature参数是模型生成结果的随机性，范围是0-2，默认是1,1.5指的是更具有创意性
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1
    )
    #4.获取响应数据
    # 接受大模型返回的数据
    ai_response = response.choices[0].message.content
    logging.info(f"AI大模型返回的数据:{ai_response}")
    #5.更新消息列表,将AI大模型返回的数据保存到会话文件中
    messages.pop (0) #删除列表中的第一个元素(提示词)
    messages.append({"role":"assistant","content":ai_response})
    session_data["messages"] = messages
    logging.info("更新后的会话信息：",messages)
    #6.保存消息列表到json文件中
    with open(session_path,"w",encoding="utf-8") as f:
        json.dump(session_data,f,ensure_ascii=False,indent=2)
    #6.返回响应的数据
    return APIresponse(code=200,message="AI会话请求相应成功",data=ai_response)

#获取会话列表
@app.get("/api/sessions")
def get_sessions()->APIresponse:  #指定返回值的数据类型为APIresponse对象
    logging.info("获取会话列表")
    #1.获取sessions目录下的所有文件名OS.listdir的作用是获取指定目录下的所有文件名
    session_files = os.listdir("sessions")
    #根据文件名获取文件名中的会话ID，遍历文件名，根据.切割获取第一个元素，获得会话ID
    session_ids = [file.split(".")[0] for file in session_files]
    #对会话列表进行倒序排序
    session_ids.sort(reverse=True)
    #将获取到的数据返回给客户端
    return APIresponse(code=200,message="获取会话列表成功",data=session_ids)


#加载指定会话(路径参数/api/sessions/{session_id})
@app.get("/api/sessions/{session_id}")
def get_session(session_id:str)->APIresponse:
    logging.info(f"加载会话:{session_id}")
    #1.获取会话文件名
    session_file = get_session_file_name(session_id)

    #2.加载会话文件
    with open(session_file,"r",encoding="utf-8") as f:
        session_data = json.load(f)

    #3.将加载到的会话数据返回给客户端
    return APIresponse(code=200,message="获取会话成功",data=session_data)

#删除指定会话
@app.delete("/api/sessions/{session_id}")
def delete_session(session_id:str)->APIresponse:
    logging.info(f"删除会话:{session_id}")
    session_file = get_session_file_name(session_id)
    if os.path.exists(session_file):
        os.remove(session_file)
    return APIresponse(code=200,message="删除会话成功",data=None)

#异常处理器：可以借助FastAPI的异常处理器，当发生异常时，异常处理器会自动调用指定的函数进行异常处理,然后向客户端响应指定的错误信息
@app.exception_handler(Exception)       #exception指的是异常类型，将会捕获所有的异常类型
def exception_handler(request:Request,exc:Exception):
    logging.error(f"处理异常，请求路径：{request.url},异常信息：{exc}")  #捕获到异常信息之后，通过logging模块输出异常信息
    # return APIresponse(code=500,message="服务器内部错误",data=None)
    return JSONResponse(content={"code":500,"message":"服务器内部错误", "data":None})


#启动函数，access_log = False 关闭访问日志
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)


