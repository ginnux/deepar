import time
import os
import yagmail
import shutil


class Logger:
    def __init__(self):
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.finishtime = "Not Finished"
        self.data = ""
        self.dirname = self.create_log_dir()
        self.plot_dirname = self.create_plot_dir()

    @staticmethod
    def copy_folder(source_folder, destination_folder):
        try:
            shutil.copytree(source_folder, destination_folder)
            print(f"文件夹 {source_folder} 已成功复制到 {destination_folder}")
        except shutil.Error as e:
            print(f"发生错误: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")

    def finish(self, config):
        self.config_to_log(config=config)

    def create_log_dir(self, timestamp=None):
        # 创建日志文件夹
        if not os.path.exists("log"):
            os.makedirs("log")
        if timestamp is None:
            timestamp = self.timestamp
            if not os.path.exists(f"log/{timestamp}"):
                os.makedirs(f"log/{timestamp}")

        if os.path.exists(f"log/{timestamp}"):
            return f"log/{timestamp}"
        else:
            raise Exception(f"Failed to create log directory: log/{timestamp}")

    def create_plot_dir(self, timestamp=None):
        # 创建日志文件夹
        if timestamp is None:
            timestamp = self.timestamp

        if not os.path.exists(f"log/{timestamp}/plot"):
            os.makedirs(f"log/{timestamp}/plot")

        if os.path.exists(f"log/{timestamp}/plot"):
            return f"log/{timestamp}/plot"
        else:
            raise Exception(f"Failed to create log directory: log/{timestamp}/plot")

    @staticmethod
    def get_timestamp():
        return time.strftime("%Y-%m-%d_%H-%M-%S")

    def config_to_log(self, config, timestamp=None):
        # 获取当前时间的格式化字符串

        if timestamp is None:
            timestamp = self.timestamp

        # 构造文件名
        filename = f"config_{timestamp}.txt"
        pathname = self.dirname + "/" + filename

        data = ""
        for key, value in config.items():
            data += f"{key}:{value}\n"

        # 保存文件
        with open(pathname, "w") as file:
            file.write(data)

    def plot_to_log_name(self, timestamp=None, file_format="pdf"):
        if timestamp is None:
            timestamp = self.timestamp
        filename = f"plot_{timestamp}" + "." + file_format
        pathname = self.dirname + "/" + filename
        return pathname

    def evaluate_to_log_name(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamp
        filename = f"evaluate_{timestamp}.csv"
        pathname = self.dirname + "/" + filename
        return pathname

    def report_to_email(self, config, timestamp=None):
        # 连接服务器
        # 用户名、授权码、服务器地址
        yag_server = yagmail.SMTP(
            user="1246150935@qq.com", password="vwcegrknoepxjgae", host="smtp.qq.com"
        )
        # 发送对象列表
        email_to = [
            "coldwarrior@163.com",
        ]
        email_title = "程序调试报告 at " + self.timestamp

        data = ""
        for key, value in config.items():
            data += f"{key}:{value}\n"

        email_content = data
        # 附件列表
        email_attachments = [self.plot_to_log_name(), self.evaluate_to_log_name()]

        # 发送邮件
        yag_server.send(email_to, email_title, email_content, email_attachments)
        # yag_server.send(email_to, email_title, email_content)

    def save_to_disk(self):
        dirname = self.dirname
        self.copy_folder(
            source_folder=dirname,
            destination_folder="/root/autodl-fs/tcn_" + self.dirname,
        )


def create_netplot_dir(timestamp=None):
    # 创建日志文件夹
    if timestamp is None:
        raise Exception("timestamp is None")

    if not os.path.exists(f"log/{timestamp}"):
        raise Exception(f"log/{timestamp} does not exist")

    if not os.path.exists(f"log/{timestamp}/netplot"):
        os.makedirs(f"log/{timestamp}/netplot")

    if os.path.exists(f"log/{timestamp}/netplot"):
        return f"log/{timestamp}/netplot"
    else:
        raise Exception(f"Failed to create log directory: log/{timestamp}/netplot")


def create_replot_dir(timestamp=None):
    # 创建日志文件夹
    if timestamp is None:
        raise Exception("timestamp is None")

    if not os.path.exists(f"log/{timestamp}"):
        raise Exception(f"log/{timestamp} does not exist")

    if not os.path.exists(f"log/{timestamp}/replot"):
        os.makedirs(f"log/{timestamp}/replot")

    if os.path.exists(f"log/{timestamp}/replot"):
        return f"log/{timestamp}/replot"
    else:
        raise Exception(f"Failed to create log directory: log/{timestamp}/replot")


class Experiment_Agent:
    def __init__(self, config):
        self.config = config
        self.exp_id = 1

        if not os.path.exists("experiment"):
            os.makedirs("experiment")

        i = 1
        while os.path.exists(f"experiment/exp_{i}.txt"):
            i += 1
        self.exp_id = i - 1

    def init_new_exp(self):
        # 创建一个记录实验的txt文件
        self.exp_id += 1

        with open(f"experiment/exp_{self.exp_id}.txt", "w") as file:
            # 写入文字到文件中
            file.write("times:0\n")

            data = ""
            for key, value in self.config.items():
                data += f"{key}:{value}\n"

            file.write(data)

    def run_exp(self, timestamp=None):
        with open(f"experiment/exp_{self.exp_id}.txt", "r") as file:
            # 读取文件中的内容
            data = file.readlines()

        times = int(data[0].split(":")[1])
        times += 1
        data[0] = f"times:{times}\n"
        data.append(f"timestamp:{timestamp}\n")

        print(data)

        with open(f"experiment/exp_{self.exp_id}.txt", "w") as file:
            # 写入文字到文件中
            file.writelines(data)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S")
