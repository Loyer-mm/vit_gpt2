import json
import os
#将val_list.json中的image_id补齐,返回到val_list1.json

# 你的文件路径
val_list_path = "data/coco2014/annotations/val_list.json"

print("开始补齐 image_id...")

with open(val_list_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

fixed_count = 0
for item in data:
    # 从文件名中提取数字 ID
    # 例如: "COCO_val2014_000000184613.jpg" -> 184613
    filename = item['image']
    try:
        # 移除前缀和后缀，转化为整数
        # 假设文件名格式固定为 COCO_val2014_000000xxxxxx.jpg
        core_id_str = filename.split('_')[-1].split('.')[0]
        image_id = int(core_id_str)
        
        item['image_id'] = image_id
        fixed_count += 1
    except Exception as e:
        print(f"处理文件名 {filename} 时出错: {e}")

# 保存更新后的文件
with open(val_list_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"修复完成！共处理 {fixed_count} 条数据。现在你可以重新运行 inference_fixed.py 了。")