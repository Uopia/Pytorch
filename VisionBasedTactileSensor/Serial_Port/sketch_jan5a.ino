String readString; // 用于存储接收到的数据
int counter = 1;   // 用于累加的计数器
bool receivedOk = false; // 标志，用于标记是否收到"okS"

void setup() {
  Serial.begin(9600); // 初始化串口通信，波特率设置为9600
}

void loop() {
  if (!receivedOk) {
    // 只有在没有收到"okS"时才发送"StartS"
    Serial.println("tartS");
    delay(500); // 稍作延迟，防止过快发送
  }

  if (Serial.available()) {
    readString = Serial.readStringUntil('S'); // 读取到 'S' 为止的字符串

    // 检查接收到的字符串是否为 "ok"
    if (readString == "ok") {
      receivedOk = true; // 设置标志为true，表示已收到"okS"
      delay(5000); // 等待 1 秒
      sendCounterString();
      counterIncrement();
    }
  }

  if (receivedOk && !Serial.available()) {
    // 如果收到过"okS"且当前没有新数据，开始重发逻辑
    unsigned long startTime = millis();
    while (!Serial.available() && millis() - startTime < 1000) {
      // 等待响应或超时
    }

    if (!Serial.available()) {
      // 如果超时且没有收到新的"okS"，重发当前计数器的值
      sendCounterString();
      startTime = millis(); // 重置计时器
    } else {
      // 如果收到新的"okS"
      readString = Serial.readStringUntil('S');
      if (readString == "ok") {
        counterIncrement();
      }
    }
  }

  delay(100); // 稍作延迟，防止CPU占用过高
}

void sendCounterString() {
  Serial.print("A");
  if (counter < 10) {
    Serial.print("00");
  } else if (counter < 100) {
    Serial.print("0");
  }
  Serial.print(counter);
  Serial.println("S");
}

void counterIncrement() {
  counter++; // 累加计数器
  if (counter > 100) {
    counter = 1; // 如果超过100，则重置计数器
  }
}
