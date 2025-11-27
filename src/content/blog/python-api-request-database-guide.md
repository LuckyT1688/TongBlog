---
title: 从API文档到数据入库：Python接口开发实战
published: 2025-11-27T14:30:00
description: 手把手教你看懂API文档、构造请求参数、解析响应数据并存入数据库，零基础也能上手的完整流程
image: ""
tags: [Python, API, 数据库, 爬虫]
category: 技术教程
draft: false
---

## 起因

上周接到一个需求：公司买了天气数据服务，每天要把全国主要城市的天气数据拉回来存数据库，供业务系统调用。给了我一份PDF格式的API文档，让我搞定。

说实话，刚开始看文档的时候有点懵。什么请求头、签名算法、时间戳验证……但写完之后发现，这事儿其实就三步：

1. 看懂文档，知道怎么发请求
2. 发请求拿到数据，解析成能用的格式
3. 把数据塞进数据库

## 第一步：读懂API文档

拿到文档先别慌，重点关注这几个部分：

### 1.1 基础信息

```
接口地址：https://api.weather.com/v1/forecast
请求方式：GET
认证方式：API Key
```

这三个最关键。地址告诉你往哪发请求，方式告诉你用GET还是POST，认证方式决定了你怎么证明"我有权限访问"。

### 1.2 请求参数

文档上写的参数表格大概长这样：

| 参数名 | 类型 | 必填 | 说明 | 示例 |
|--------|------|------|------|------|
| city | string | 是 | 城市代码 | 110000 |
| date | string | 否 | 查询日期 | 2025-11-27 |
| api_key | string | 是 | 密钥 | your_key_here |

必填的一定要带，可选的看需求。城市代码这种一般文档最后会附一个码表。

### 1.3 返回结果

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "city": "北京",
    "date": "2025-11-27",
    "temperature": {
      "max": 12,
      "min": -2
    },
    "weather": "晴",
    "wind": "北风3级",
    "aqi": 45
  }
}
```

`code` 字段一般表示请求状态，200是成功，其他都是失败。`data` 里面才是真正有用的数据。

## 第二步：写代码发请求

### 2.1 安装依赖

```bash
pip install requests pymysql pandas
```

`requests` 负责发HTTP请求，`pymysql` 连接MySQL数据库，`pandas` 用来处理数据（可选，但确实好用）。

### 2.2 最简单的请求

先写个最简单的，能通就行：

```python
import requests

url = "https://api.weather.com/v1/forecast"
params = {
    "city": "110000",  # 北京
    "api_key": "你的密钥"
}

response = requests.get(url, params=params)
print(response.text)
```

运行一下，如果打印出JSON数据，说明通了。如果报错，看报错信息：
- `401 Unauthorized`：密钥不对或过期
- `404 Not Found`：地址写错了
- `500 Internal Server Error`：服务端挂了，不是你的问题

### 2.3 处理返回数据

拿到数据后要解析成Python能用的格式：

```python
def fetch_weather(city_code):
    """获取天气数据"""
    url = "https://api.weather.com/v1/forecast"
    params = {
        "city": city_code,
        "api_key": "你的密钥"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 4xx或5xx会抛异常
        
        result = response.json()
        
        # 检查业务状态码
        if result.get("code") != 200:
            print(f"接口返回错误：{result.get('message')}")
            return None
            
        return result.get("data")
        
    except requests.exceptions.Timeout:
        print("请求超时")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        return None
```

这里做了几件事：
- 加了超时时间，防止接口挂了一直等
- 用 `raise_for_status()` 检查HTTP状态码
- 单独检查业务状态码（有些接口HTTP返回200但业务失败）
- 异常处理，出错不至于程序崩溃

### 2.4 批量获取多个城市

实际场景肯定要查多个城市：

```python
# 城市代码映射表（实际从文档或数据库读）
CITIES = {
    "110000": "北京",
    "310000": "上海",
    "440100": "广州",
    "440300": "深圳"
}

def fetch_all_weather():
    """批量获取天气数据"""
    weather_list = []
    
    for city_code, city_name in CITIES.items():
        print(f"正在获取 {city_name} 的天气...")
        data = fetch_weather(city_code)
        
        if data:
            # 补充城市代码，方便后续入库
            data["city_code"] = city_code
            weather_list.append(data)
            
        # 礼貌性延迟，别把人家接口打爆了
        time.sleep(0.5)
    
    return weather_list
```

注意那个 `time.sleep(0.5)`，很多API都有频率限制，比如每秒最多10次请求。不加延迟可能被封IP。

## 第三步：数据入库

### 3.1 设计数据表

根据返回的数据结构设计表：

```sql
CREATE TABLE weather_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    city_code VARCHAR(10) NOT NULL,
    city_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    weather VARCHAR(50),
    temp_max INT,
    temp_min INT,
    wind VARCHAR(50),
    aqi INT,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_city_date (city_code, date)  -- 防止重复插入
);
```

`UNIQUE KEY` 很重要，避免同一天同一城市的数据插入多次。

### 3.2 写入数据库

```python
import pymysql
from datetime import datetime

class WeatherDB:
    """天气数据库操作类"""
    
    def __init__(self, host, user, password, database):
        self.conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
        self.cursor = self.conn.cursor()
    
    def save_weather(self, weather_data):
        """保存天气数据"""
        sql = """
            INSERT INTO weather_data 
            (city_code, city_name, date, weather, temp_max, temp_min, wind, aqi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                weather = VALUES(weather),
                temp_max = VALUES(temp_max),
                temp_min = VALUES(temp_min),
                wind = VALUES(wind),
                aqi = VALUES(aqi)
        """
        
        values = (
            weather_data.get("city_code"),
            weather_data.get("city"),
            weather_data.get("date"),
            weather_data.get("weather"),
            weather_data.get("temperature", {}).get("max"),
            weather_data.get("temperature", {}).get("min"),
            weather_data.get("wind"),
            weather_data.get("aqi")
        )
        
        try:
            self.cursor.execute(sql, values)
            self.conn.commit()
            print(f"✓ {weather_data.get('city')} 数据已保存")
        except Exception as e:
            self.conn.rollback()
            print(f"✗ 保存失败：{e}")
    
    def close(self):
        """关闭连接"""
        self.cursor.close()
        self.conn.close()
```

`ON DUPLICATE KEY UPDATE` 是MySQL的特性，如果遇到唯一键冲突就执行更新而不是报错。适合每天定时跑的任务。

### 3.3 完整流程

把前面的代码串起来：

```python
import time
import requests
import pymysql
from datetime import datetime

# 城市映射
CITIES = {
    "110000": "北京",
    "310000": "上海",
    "440100": "广州",
    "440300": "深圳"
}

def fetch_weather(city_code):
    """获取天气数据"""
    url = "https://api.weather.com/v1/forecast"
    params = {
        "city": city_code,
        "api_key": "your_api_key_here"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result.get("code") != 200:
            print(f"接口错误：{result.get('message')}")
            return None
            
        return result.get("data")
    except Exception as e:
        print(f"请求失败：{e}")
        return None

class WeatherDB:
    """数据库操作"""
    
    def __init__(self, host, user, password, database):
        self.conn = pymysql.connect(
            host=host, user=user, password=password,
            database=database, charset='utf8mb4'
        )
        self.cursor = self.conn.cursor()
    
    def save_weather(self, data):
        """保存数据"""
        sql = """
            INSERT INTO weather_data 
            (city_code, city_name, date, weather, temp_max, temp_min, wind, aqi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                weather=VALUES(weather), temp_max=VALUES(temp_max),
                temp_min=VALUES(temp_min), wind=VALUES(wind), aqi=VALUES(aqi)
        """
        values = (
            data.get("city_code"),
            data.get("city"),
            data.get("date"),
            data.get("weather"),
            data.get("temperature", {}).get("max"),
            data.get("temperature", {}).get("min"),
            data.get("wind"),
            data.get("aqi")
        )
        
        try:
            self.cursor.execute(sql, values)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"入库失败：{e}")
            return False
    
    def close(self):
        self.cursor.close()
        self.conn.close()

def main():
    """主函数"""
    print("=" * 50)
    print(f"开始采集天气数据 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 连接数据库
    db = WeatherDB(
        host="localhost",
        user="root",
        password="your_password",
        database="weather_db"
    )
    
    success_count = 0
    
    # 遍历城市
    for city_code, city_name in CITIES.items():
        print(f"\n正在获取 {city_name} 的数据...")
        
        # 请求接口
        weather_data = fetch_weather(city_code)
        if not weather_data:
            continue
        
        # 补充城市代码
        weather_data["city_code"] = city_code
        
        # 存入数据库
        if db.save_weather(weather_data):
            success_count += 1
            print(f"✓ {city_name} 数据已保存")
        
        # 延迟防止频率限制
        time.sleep(0.5)
    
    db.close()
    
    print("\n" + "=" * 50)
    print(f"采集完成！成功 {success_count}/{len(CITIES)} 条")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

运行效果：

```
==================================================
开始采集天气数据 - 2025-11-27 14:30:15
==================================================

正在获取 北京 的数据...
✓ 北京 数据已保存

正在获取 上海 的数据...
✓ 上海 数据已保存

正在获取 广州 的数据...
✓ 广州 数据已保存

正在获取 深圳 的数据...
✓ 深圳 数据已保存

==================================================
采集完成！成功 4/4 条
==================================================
```

## 进阶优化

### 4.1 配置文件管理

别把密钥写死在代码里，用配置文件：

```python
# config.ini
[api]
base_url = https://api.weather.com/v1
api_key = your_key_here

[database]
host = localhost
user = root
password = your_password
database = weather_db
```

读取配置：

```python
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

API_KEY = config.get('api', 'api_key')
DB_HOST = config.get('database', 'host')
```

### 4.2 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('weather.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 使用
logger.info(f"开始获取 {city_name} 数据")
logger.error(f"请求失败：{e}")
```

日志会同时输出到文件和控制台，方便排查问题。

### 4.3 定时任务

Linux上用crontab：

```bash
# 每天早上8点执行
0 8 * * * /usr/bin/python3 /path/to/weather.py >> /path/to/weather.log 2>&1
```

Windows上可以用任务计划程序，或者代码里用 `schedule` 库：

```python
import schedule
import time

def job():
    main()

# 每天8点执行
schedule.every().day.at("08:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 4.4 异常重试

接口偶尔不稳定，加个重试机制：

```python
def fetch_weather_with_retry(city_code, max_retries=3):
    """带重试的获取天气"""
    for i in range(max_retries):
        data = fetch_weather(city_code)
        if data:
            return data
        
        if i < max_retries - 1:
            wait_time = (i + 1) * 2  # 递增等待时间
            print(f"重试 {i+1}/{max_retries}，{wait_time}秒后继续...")
            time.sleep(wait_time)
    
    return None
```

### 4.5 数据验证

入库前检查数据完整性：

```python
def validate_weather_data(data):
    """验证数据"""
    required_fields = ["city", "date", "weather"]
    
    for field in required_fields:
        if not data.get(field):
            return False, f"缺少字段：{field}"
    
    # 检查温度范围
    temp_max = data.get("temperature", {}).get("max")
    temp_min = data.get("temperature", {}).get("min")
    
    if temp_max is not None and temp_min is not None:
        if temp_max < temp_min:
            return False, "最高温不能低于最低温"
        if temp_max > 60 or temp_min < -60:
            return False, "温度超出合理范围"
    
    return True, ""

# 使用
is_valid, error_msg = validate_weather_data(weather_data)
if not is_valid:
    print(f"数据验证失败：{error_msg}")
    continue
```

## 常见坑点

### 5.1 字符编码问题

如果接口返回中文乱码：

```python
response.encoding = 'utf-8'  # 强制指定编码
data = response.json()
```

### 5.2 SSL证书验证

有些内网接口证书不合规，会报SSL错误：

```python
# 临时方案：关闭验证（生产环境慎用）
response = requests.get(url, params=params, verify=False)

# 更好的方案：指定证书路径
response = requests.get(url, params=params, verify='/path/to/cert.pem')
```

### 5.3 JSON解析失败

接口偶尔返回非JSON格式（比如HTML错误页）：

```python
try:
    result = response.json()
except ValueError:
    print(f"返回内容不是JSON：{response.text[:200]}")
    return None
```

### 5.4 数据库连接超时

长时间运行的脚本，数据库连接可能断开：

```python
def reconnect_if_needed(db):
    """检查并重连数据库"""
    try:
        db.conn.ping(reconnect=True)
    except:
        db.conn = pymysql.connect(...)
```

### 5.5 时区问题

有些接口返回UTC时间，入库前要转成本地时间：

```python
from datetime import datetime, timezone, timedelta

# UTC转东八区
utc_time = datetime.fromisoformat(data["timestamp"])
local_time = utc_time + timedelta(hours=8)
```

## 总结

整个流程就是：

1. **读文档** → 找到接口地址、参数、认证方式
2. **发请求** → 用 `requests` 库，处理好异常
3. **解析数据** → 检查状态码，提取需要的字段
4. **入库** → 设计好表结构，用 `pymysql` 写入

第一次写可能觉得步骤多，但套路都一样。以后遇到新接口，把这套代码改改参数和字段映射就能用。

代码放GitHub上了：[weather-api-demo](https://github.com/yourname/weather-api-demo)（自己建个仓库替换链接）

有问题欢迎留言，看到会回。
