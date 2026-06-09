import notifee from '@notifee/react-native';
import Geolocation from '@react-native-community/geolocation';
import mqtt from 'mqtt';
import React, { useEffect, useState, useRef } from 'react';
import { NativeModules } from 'react-native';
import * as RNFS from '@dr.pogodin/react-native-fs';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  Button,
  ScrollView,
  ActivityIndicator,
  PermissionsAndroid,
  Platform,
  TouchableOpacity,
  TextInput,
  Alert,
  AppRegistry,
} from 'react-native';

// ==================== 文件路径 ====================
const EPICENTER_FILE = `${RNFS.DocumentDirectoryPath}/epicenter.json`;
const CONFIG_FILE = `${RNFS.DocumentDirectoryPath}/config.json`;

// ==================== 类型定义 ====================
interface EarthquakeAlert {
  level: number;
  confidence: number;
  magnitude: number;
  rawText: string;
}

// ==================== 主组件 ====================
const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  // 震中来源：'auto' = MQTT自动获取, 'manual' = 手动输入
  const [epicenterSource, setEpicenterSource] = useState<'auto' | 'manual'>('manual');
  const [epicenter, setEpicenter] = useState<{ lat: number; lon: number } | null>(null);
  const epicenterRef = useRef<{ lat: number; lon: number } | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [showAlert, setShowAlert] = useState(false);
  const [alertData, setAlertData] = useState({
    distance: 0,
    magnitude: 0,
    eta: 0,
    level: 0,
    confidence: 0,
    intensity: 0,
    epicenterSource: '',
  });

  // 震中手动输入
  const [inputLat, setInputLat] = useState('');
  const [inputLon, setInputLon] = useState('');

  // 烈度阈值
  const [minIntensity, setMinIntensity] = useState(1);
  const minIntensityRef = useRef(1);
  const [inputMinIntensity, setInputMinIntensity] = useState('1');

  // 添加日志
  const addLog = (text: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${text}`, ...prev.slice(0, 9)]);
  };

  // ======== 启动后台服务的函数 ========
  const startBackgroundTask = () => {
    try {
      AppRegistry.startHeadlessTask('EarthquakeBackgroundTask', {
        taskName: 'EarthquakeBackgroundTask',
        taskData: { startTime: Date.now() },
      });
      addLog('✅ 后台任务已启动');
    } catch (error) {
      addLog(`❌ 启动后台任务失败: ${error}`);
    }
  };

  // ===== 文件存储函数 =====
  const saveEpicenterToFile = async (lat: number, lon: number, source: string) => {
    try {
      const data = { lat, lon, source };
      await RNFS.writeFile(EPICENTER_FILE, JSON.stringify(data), 'utf8');
      addLog(`💾 震中已保存: ${lat}, ${lon} (来源: ${source})`);
      return true;
    } catch (error) {
      addLog(`❌ 保存文件失败: ${error}`);
      return false;
    }
  };

  const loadEpicenterFromFile = async () => {
    try {
      const exists = await RNFS.exists(EPICENTER_FILE);
      if (!exists) return null;
      const content = await RNFS.readFile(EPICENTER_FILE, 'utf8');
      const data = JSON.parse(content);
      return data;
    } catch (error) {
      addLog(`❌ 读取震中文件失败: ${error}`);
      return null;
    }
  };

  // ===== 配置文件读写 =====
  const saveConfigToFile = async (minIntensity: number) => {
    try {
      const data = { minIntensity };
      await RNFS.writeFile(CONFIG_FILE, JSON.stringify(data), 'utf8');
      addLog(`💾 配置已保存: 最小烈度 ${minIntensity}`);
      return true;
    } catch (error) {
      addLog(`❌ 保存配置失败: ${error}`);
      return false;
    }
  };

  const loadConfigFromFile = async () => {
    try {
      const exists = await RNFS.exists(CONFIG_FILE);
      if (!exists) {
        addLog('⚠️ 没有找到配置文件，使用默认值');
        return { minIntensity: 1 };
      }
      const content = await RNFS.readFile(CONFIG_FILE, 'utf8');
      const data = JSON.parse(content);
      addLog(`📖 读取配置: 最小烈度 ${data.minIntensity}`);
      return data;
    } catch (error) {
      addLog(`❌ 读取配置失败: ${error}`);
      return { minIntensity: 1 };
    }
  };

  // ===== 请求位置权限 =====
  const requestLocationPermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const hasPermission = await PermissionsAndroid.check(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
        );
        if (hasPermission) return true;
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
          {
            title: '需要位置权限',
            message: '地震预警需要您的位置来计算您到震中的距离',
            buttonNeutral: '稍后询问',
            buttonNegative: '拒绝',
            buttonPositive: '允许',
          }
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        addLog(`❌ 权限请求出错: ${err}`);
        return false;
      }
    }
    return true;
  };

  // ===== 加载保存的数据 =====
  useEffect(() => {
    const loadAll = async () => {
      const saved = await loadEpicenterFromFile();
      if (saved) {
        setEpicenter({ lat: saved.lat, lon: saved.lon });
        epicenterRef.current = { lat: saved.lat, lon: saved.lon };
        setEpicenterSource(saved.source || 'manual');
        setInputLat(saved.lat.toString());
        setInputLon(saved.lon.toString());
        addLog(`📍 从文件加载震中: ${saved.lat}, ${saved.lon} (来源: ${saved.source || 'manual'})`);
      } else {
        const defaultEpicenter = { lat: 39.9042, lon: 116.4074 };
        setEpicenter(defaultEpicenter);
        epicenterRef.current = defaultEpicenter;
        setInputLat('39.9042');
        setInputLon('116.4074');
        addLog('📍 使用默认震中: 北京');
      }
      const config = await loadConfigFromFile();
      setMinIntensity(config.minIntensity);
      minIntensityRef.current = config.minIntensity;
      setInputMinIntensity(config.minIntensity.toString());
    };
    loadAll();
  }, []);

  // ===== 保存震中位置 =====
  const saveEpicenter = async () => {
    const lat = parseFloat(inputLat);
    const lon = parseFloat(inputLon);
    if (isNaN(lat) || isNaN(lon)) {
      Alert.alert('错误', '请输入有效的经纬度');
      return;
    }
    const success = await saveEpicenterToFile(lat, lon, 'manual');
    if (success) {
      setEpicenter({ lat, lon });
      epicenterRef.current = { lat, lon };
      setEpicenterSource('manual');
      addLog(`📍 震中已手动更新: ${lat}, ${lon}`);
    }
  };

  // ===== 保存配置 =====
  const saveConfig = async () => {
    const val = parseInt(inputMinIntensity);
    if (isNaN(val) || val < 0 || val > 12) {
      Alert.alert('错误', '请输入0-12之间的烈度');
      return;
    }
    const success = await saveConfigToFile(val);
    if (success) {
      setMinIntensity(val);
      minIntensityRef.current = val;
      addLog(`⚙️ 最小预警烈度已设为 ${val} 度`);
    }
  };

  // ===== 获取用户当前位置 =====
  const getCurrentLocation = (): Promise<{ lat: number; lon: number }> => {
    return new Promise(async (resolve, reject) => {
      const hasPermission = await requestLocationPermission();
      if (!hasPermission) {
        reject('位置权限被拒绝');
        return;
      }
      Geolocation.getCurrentPosition(
        (pos) => {
          resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        },
        (err) => {
          let errorMsg = '未知错误';
          switch (err.code) {
            case 1: errorMsg = '用户拒绝了位置权限'; break;
            case 2: errorMsg = '无法获取位置（网络或GPS问题）'; break;
            case 3: errorMsg = '定位超时'; break;
          }
          addLog(`❌ 位置获取失败: ${errorMsg}`);
          reject(err.message);
        },
        { enableHighAccuracy: true, timeout: 30000, maximumAge: 0 }
      );
    });
  };

  const [mqttStatus, setMqttStatus] = useState('disconnected');
  const [mqttClient, setMqttClient] = useState<any>(null);

  // ===== MQTT 连接 =====
  const connectMQTT = () => {
    const MQTT_CONFIG = {
      username: 'User',
      password: '1234567890',
      clientId: `EQNew_${Math.random().toString(36).substring(7)}`,
      uri: 'wss://w8381dbc.ala.cn-hangzhou.emqxsl.cn:8084/mqtt',
    };

    addLog(`🔄 正在连接 MQTT...`);

    try {
      const client = mqtt.connect(MQTT_CONFIG.uri, {
        clientId: MQTT_CONFIG.clientId,
        username: MQTT_CONFIG.username,
        password: MQTT_CONFIG.password,
        clean: true,
        keepalive: 60,
        reconnectPeriod: 5000,
        connectTimeout: 30 * 1000,
        rejectUnauthorized: false,
      });

      client.on('connect', () => {
        setMqttStatus('connected');
        addLog('✅ MQTT 连接成功');
        client.subscribe('earthquake/alert', { qos: 0 });
        startBackgroundTask();
      });

      // ======== 发送通知的函数 ========
      const sendNotification = async (intensity: number, distance: number, eta: number) => {
        try {
          const settings = await notifee.requestPermission();
          if (settings.authorizationStatus < 1) {
            addLog('❌ 用户拒绝了通知权限');
            return;
          }
          const channelId = await notifee.createChannel({
            id: 'earthquake',
            name: '地震预警',
            sound: 'default',
            vibration: true,
            importance: 4,
          });
          await notifee.displayNotification({
            title: `⚠️ 地震预警 ${intensity}度`,
            body: `距离${distance.toFixed(1)}km，${eta.toFixed(0)}秒后到达`,
            android: {
              channelId,
              pressAction: { id: 'default' },
              visibility: 1,
              asForegroundService: true,
              color: '#FF6B6B',
            },
            ios: {
              sound: 'default',
              badgeCount: 1,
              interruptionLevel: 'timeSensitive',
            },
          });
          addLog('✅ 系统通知已发送');
        } catch (error) {
          addLog(`❌ 发送通知失败: ${error}`);
        }
      };

      // ===== 收到 MQTT 消息 =====
            // ===== 收到 MQTT 消息 =====
      client.on('message', async (topic: string, message: Buffer) => {
        try {
          const rawMsg = message.toString();
          addLog(`📨 收到: ${rawMsg.substring(0, 200)}`);

          const lastOpen = rawMsg.lastIndexOf('{');
          const lastClose = rawMsg.lastIndexOf('}');
          if (lastOpen === -1 || lastClose === -1 || lastClose <= lastOpen) {
            addLog('❌ 未找到完整的 JSON');
            return;
          }

          let jsonStr = rawMsg.substring(lastOpen, lastClose + 1);
          if (jsonStr.endsWith('}}')) {
            jsonStr = jsonStr.slice(0, -1);
          }

          const data = JSON.parse(jsonStr);

          // ==========================================
          // 🔥 自动提取震中坐标
          // ==========================================
          if (data.gwLat !== undefined && data.gwLng !== undefined) {
            const gwLat = parseFloat(data.gwLat);
            const gwLng = parseFloat(data.gwLng);
            if (!isNaN(gwLat) && !isNaN(gwLng) && gwLat !== 0 && gwLng !== 0) {
              setEpicenter({ lat: gwLat, lon: gwLng });
              epicenterRef.current = { lat: gwLat, lon: gwLng };
              setEpicenterSource('auto');
              await saveEpicenterToFile(gwLat, gwLng, 'auto');
              addLog(`📍 自动获取震中: ${gwLat.toFixed(4)}, ${gwLng.toFixed(4)}`);
            }
          }

          // ==========================================
          // 根据 type 字段判断消息类型
          // ==========================================
          let confidence = 0;
          let intensity = 0;

          if (data.type === 'trigger') {
  // ===== 区分状态机通知和4G告警 =====

  
  if (data.node !== undefined) {
    // 4G 模式报警
    confidence = parseFloat(data.confidence) || 0;
    intensity = parseInt(data.intensity) || 5;
    addLog(`📡 4G报警: node=${data.node}, conf=${confidence}, int=${intensity}`);
  } else {
    addLog(`⚠️ 未知trigger格式`);
    return;
  }
  
} else if (data.text) {
  // 钉钉格式
  const text = data.text;
  const confidenceMatch = text.match(/置信度.*?([\d.]+)/);
  const intensityMatch = text.match(/烈度.*?(\d+)/);
  confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : 0;
  intensity = intensityMatch ? parseInt(intensityMatch[1]) : 0;
  addLog(`📡 钉钉报警: conf=${confidence}, int=${intensity}`);
  
} else if (data.type === 'pca_forward') {
  addLog(`📊 PCA数据转发 → 节点${data.node}, 跳过`);
  return;
} else if (data.type === 'presence') {
  addLog(`📍 人员到场 → 节点${data.node}, 跳过`);
  return;
} else if (data.type === 'state_change') {
  // 状态机变化（MQTT格式）
  if (data.level === 'NONE') {
    addLog(`✅ 预警解除`);
  } else {
    addLog(`📢 状态变化: ${data.level}, 置信度: ${data.prob || 0}`);
  }
  return;
} else {
  addLog(`⚠️ 未知消息类型: ${data.type || '无type字段'}`);
  return;
}
           

          if (intensity < minIntensityRef.current) {
            addLog(`⏭️ 烈度 ${intensity} 低于阈值 ${minIntensityRef.current}，不预警`);
            return;
          }

          if (!epicenterRef.current) {
            addLog('⚠️ 请先设置震中位置');
            return;
          }

          addLog('📍 正在获取您的位置...');
          const userLoc = await getCurrentLocation();
          addLog(`📍 当前位置: ${userLoc.lat.toFixed(4)}, ${userLoc.lon.toFixed(4)}`);

          const distance = calculateDistance(
            userLoc.lat, userLoc.lon,
            epicenterRef.current.lat, epicenterRef.current.lon
          );
          const eta = distance / 3.5;

          addLog(`⚠️ 震中距 ${distance.toFixed(1)}km, 预计 ${eta.toFixed(0)} 秒后到达`);

          setAlertData({
    distance,
    magnitude: 0,
    eta,
    level: intensity,
    confidence,
    intensity,
    epicenterSource: 'auto',  // ← 直接写 'auto'
});
          setShowAlert(true);
          await sendNotification(intensity, distance, eta);
          setTimeout(() => setShowAlert(false), 5000);

        } catch (e: any) {
          addLog(`❌ 解析失败: ${e.message}`);
        }
      });

      client.on('error', (err: any) => {
        addLog(`❌ MQTT 错误: ${err.message}`);
      });

      client.on('close', () => {
        addLog('❌ MQTT 断开');
        setMqttStatus('disconnected');
      });

      setMqttClient(client);
    } catch (error: any) {
      addLog(`❌ MQTT 连接失败: ${error.message}`);
    }
  };

  // ===== 距离计算 =====
  const calculateDistance = (
    lat1: number, lon1: number,
    lat2: number, lon2: number
  ) => {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  // ===== 初始化 =====
  useEffect(() => {
    const init = async () => {
      addLog('🚀 应用启动中...');
      connectMQTT();
      setIsLoading(false);
      addLog('✅ 初始化完成');
    };
    init();
  }, []);

  // ===== 加载页面 =====
  if (isLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#FF6B6B" />
        <Text style={styles.loadingText}>正在初始化地震预警系统...</Text>
      </View>
    );
  }

  // ===== 主界面 =====
  return (
    <SafeAreaView style={styles.container}>
      {/* 弹窗 */}
      {showAlert && (
        <View style={styles.overlay}>
          <View style={styles.alertBox}>
            <Text style={styles.alertTitle}>⚠️ 地震预警 ⚠️</Text>
            <Text style={styles.alertText}>级别：{alertData.level} 级</Text>
            <Text style={styles.alertText}>置信度：{alertData.confidence/4}%</Text>
            <Text style={styles.alertText}>烈度：{alertData.intensity} 度</Text>
            <Text style={styles.alertText}>震中距：{alertData.distance.toFixed(1)} 公里</Text>
            <Text style={styles.alertText}>震中来: {alertData.epicenterSource === 'auto' ? '📡 自动获取' : '✋ 手动输入'}</Text>
            <Text style={[styles.alertText, { fontSize: 24, color: 'white' }]}>
              {alertData.eta.toFixed(0)} 秒后到达
            </Text>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: `${Math.min(100, (alertData.eta / 10) * 100)}%` }]} />
            </View>
            <Text style={styles.alertClose} onPress={() => setShowAlert(false)}>点击关闭</Text>
          </View>
        </View>
      )}

      {/* 头部 */}
      <View style={styles.header}>
        <Text style={styles.title}>🌍 地震预警</Text>
        <View style={styles.status}>
          <View style={[styles.dot, { backgroundColor: mqttStatus === 'connected' ? '#4CAF50' : '#FF6B6B' }]} />
          <Text style={styles.statusText}>{mqttStatus === 'connected' ? '已连接' : '连接中'}</Text>
        </View>
      </View>

      {/* 震中设置 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📍 震中位置</Text>
        <Text style={styles.locationText}>
          当前来源: {epicenterSource === 'auto' ? '📡 MQTT自动获取' : '✋ 手动输入'}
        </Text>

        {/* 手动输入 */}
        <TextInput
          style={styles.input}
          placeholder="纬度 (例如 39.9042)"
          value={inputLat}
          onChangeText={setInputLat}
          keyboardType="numeric"
        />
        <TextInput
          style={styles.input}
          placeholder="经度 (例如 116.4074)"
          value={inputLon}
          onChangeText={setInputLon}
          keyboardType="numeric"
        />
        <TouchableOpacity onPress={saveEpicenter} style={styles.smallButton}>
          <Text style={styles.buttonText}>💾 手动保存震中</Text>
        </TouchableOpacity>
        {epicenter && (
          <Text style={styles.locationText}>
            当前震中: {epicenter.lat.toFixed(4)}, {epicenter.lon.toFixed(4)}
          </Text>
        )}
      </View>

      {/* 预警设置 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>⚙️ 预警设置</Text>
        <Text style={styles.label}>最小预警烈度 (0-12)</Text>
        <TextInput
          style={styles.input}
          placeholder="例如 3"
          value={inputMinIntensity}
          onChangeText={setInputMinIntensity}
          keyboardType="numeric"
        />
        <TouchableOpacity onPress={saveConfig} style={styles.smallButton}>
          <Text style={styles.buttonText}>💾 保存设置</Text>
        </TouchableOpacity>
        <Text style={styles.locationText}>
          当前阈值: {minIntensity} 度 (低于此烈度不预警)
        </Text>
      </View>

      {/* 日志 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🕒 实时日志</Text>
        <ScrollView style={styles.logContainer}>
          {logs.map((log, i) => (
            <Text key={i} style={styles.logText}>{log}</Text>
          ))}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
};

// ==================== 样式 ====================
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#f5f5f5' },
  loadingText: { marginTop: 10, fontSize: 16, color: '#666' },
  header: {
    backgroundColor: '#fff',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: { fontSize: 24, fontWeight: 'bold', color: '#FF6B6B' },
  status: { flexDirection: 'row', alignItems: 'center' },
  dot: { width: 10, height: 10, borderRadius: 5, marginRight: 5 },
  statusText: { fontSize: 14, color: '#666' },
  section: {
    backgroundColor: '#fff',
    margin: 10,
    padding: 15,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  sectionTitle: { fontSize: 18, fontWeight: '600', marginBottom: 10, color: '#333' },
  label: { fontSize: 14, color: '#666', marginBottom: 5 },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 5,
    padding: 10,
    marginVertical: 5,
    fontSize: 16,
  },
  locationText: { fontSize: 14, color: '#666', marginTop: 10, textAlign: 'center' },
  logContainer: { maxHeight: 150, backgroundColor: '#fafafa', padding: 10, borderRadius: 5 },
  logText: { fontSize: 12, color: '#666', marginBottom: 3 },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.85)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999,
    elevation: 9999,
  },
  alertBox: {
    backgroundColor: '#FF6B6B',
    width: '98%',
    padding: 30,
    borderRadius: 20,
    alignItems: 'center',
    borderWidth: 3,
    borderColor: 'white',
  },
  alertTitle: { fontSize: 28, fontWeight: 'bold', color: 'white', marginBottom: 20 },
  alertText: { fontSize: 18, color: 'white', marginVertical: 5 },
  progressBar: {
    width: '100%',
    height: 10,
    backgroundColor: 'rgba(255,255,255,0.3)',
    borderRadius: 5,
    marginVertical: 15,
    overflow: 'hidden',
  },
  progressFill: { height: '100%', backgroundColor: 'white' },
  alertClose: {
    color: 'white',
    fontSize: 16,
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderWidth: 2,
    borderColor: 'white',
    borderRadius: 10,
    overflow: 'hidden',
  },
  smallButton: {
    backgroundColor: '#4CAF50',
    padding: 10,
    borderRadius: 5,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonText: { color: 'white', fontSize: 14, fontWeight: '500' },
});

export default App;