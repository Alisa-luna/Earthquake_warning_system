# data_manager.py
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import os
import json
import time


class EarthquakeDataManager:
    """地震数据自动下载管理器"""

    def __init__(self, data_dir='test_data'):
        self.client = Client("IRIS")
        self.data_dir = data_dir
        self.catalog_file = f"{data_dir}/catalog.json"

        # 创建目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 测试用例库
        self.test_cases = [
            # 烈度5度组（应该触发橙色预警）
            {
                'name': 'M5.0_intensity5',
                'event': '2019-06-22T03:40:00',
                'mag': 5.0,
                'stations': [
                    {'net': 'CI', 'sta': 'BAK', 'chan': 'BH*', 'expected_intensity': 5}
                ]
            },
            # 烈度6度组（应该触发橙色预警）
            {
                'name': 'M5.5_intensity6',
                'event': '2020-06-24T03:40:00',
                'mag': 5.5,
                'stations': [
                    {'net': 'CI', 'sta': 'BAK', 'chan': 'BH*', 'expected_intensity': 6}
                ]
            },
            # 烈度7度组（应该触发红色预警）
            {
                'name': 'M6.4_intensity7',
                'event': '2019-07-04T17:33:00',
                'mag': 6.4,
                'stations': [
                    {'net': 'CI', 'sta': 'GSC', 'chan': 'BH*', 'expected_intensity': 7}
                ]
            },
            # 无感事件（应该不触发）
            {
                'name': 'M3.0_noise',
                'event': '2019-07-10T00:00:00',
                'mag': 3.0,
                'stations': [
                    {'net': 'CI', 'sta': 'BAK', 'chan': 'BH*', 'expected_intensity': 1}
                ]
            }
        ]

    def download_all(self):
        """下载所有测试用例"""
        catalog = []

        for case in self.test_cases:
            t = UTCDateTime(case['event'])

            for station in case['stations']:
                try:
                    print(f"📥 下载 {case['name']} | {station['net']}.{station['sta']}")

                    st = self.client.get_waveforms(
                        network=station['net'],
                        station=station['sta'],
                        location="*",
                        channel=station['chan'],
                        starttime=t - 60,  # 震前1分钟
                        endtime=t + 180  # 震后3分钟
                    )

                    # 保存数据
                    filename = f"{self.data_dir}/{case['name']}_{station['net']}_{station['sta']}.mseed"
                    st.write(filename, format="MSEED")

                    # 记录到目录
                    catalog.append({
                        'test_id': f"{case['name']}_{station['net']}_{station['sta']}",
                        'filename': filename,
                        'expected_intensity': station['expected_intensity'],
                        'mag': case['mag'],
                        'event_time': case['event']
                    })

                    time.sleep(0.5)

                except Exception as e:
                    print(f"❌ 下载失败: {e}")

        # 保存目录
        with open(self.catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)

        return catalog