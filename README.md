# Router策略

## 運作方式
- **先下載t5-large model**：下載t5-large model到initial_model/t5-large。
- **運行bash檔案**：
    依照下面指令運行，可以產出order1.log於檔案包內，可以加入修改order1.sh變成其他order。
    ```bash
    nohup bash order1.sh> order1.log 2>&1 &
    ```
## 注意事項
- **model/layers.py**：
    MoEBlock的部份那邊有注意事項，需要依據是否採行新的Router策略做變動