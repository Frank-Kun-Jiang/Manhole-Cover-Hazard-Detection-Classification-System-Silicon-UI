import sqlite3

class ImageResultDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_table()

    def _connect(self):
        """建立数据库连接并初始化游标"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _create_table(self):
        """创建存储图像结果的表（如果不存在）只包含处理后图片路径和预测标签"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                processed_image_path TEXT,
                predicted_label TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def insert_img_result(self, processed_image_path, predicted_label):
        """插入处理结果到数据库，只保存处理后图片路径和预测标签"""
        self.cursor.execute('''
            INSERT INTO image_results (processed_image_path, predicted_label)
            VALUES (?, ?)
        ''', (processed_image_path, predicted_label))
        self.conn.commit()

    def query_results(self):
        """查询所有存储的处理结果"""
        self.cursor.execute('SELECT * FROM image_results')
        return self.cursor.fetchall()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
