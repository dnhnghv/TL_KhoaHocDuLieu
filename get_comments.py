import requests
import pandas as pd

# 1. Cấu hình
page_token = "EAA6U8x8RuMMBOZCITz8uZAVsuDgzFNAw1NX1QXfe1LvHeHKzGtULbQiy529GmCZA6NPZAwRUR1tw5zUKVZC5BEDSEugkkFNw8ZCmCG0vYbiIAWLVkAbfj0X71UK8JkyiqWxYJfmxIODopeZCZAXgtcjGQVFEP87kp6iUfI8GQZAQERGT9BfY8EUO4VCK2LzxraD3zjuCobk8ZCQz2G8NxjXoRau0TPF9C3XLmWCZC505bFlcOUH"
post_id = "681112348415895_122093903948898898"  # ví dụ bài "ngonn"

# 2. Hàm lấy toàn bộ comment (phân trang)
def get_all_comments(post_id, access_token):
    url = f"https://graph.facebook.com/v12.0/{post_id}/comments"
    params = {
        "access_token": access_token,
        "summary": "true",
        "filter": "stream",
        "limit": 100
    }
    all_comments = []
    while True:
        resp = requests.get(url, params=params).json()
        data = resp.get("data", [])
        all_comments.extend(data)
        # Kiểm paging
        paging = resp.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break
        # Next page không cần params
        url = next_url
        params = {}
    return all_comments, resp.get("summary", {})

# 3. Thực thi
comments, summary = get_all_comments(post_id, page_token)

# 4. Xuất ra DataFrame và CSV
df = pd.DataFrame([{
    "stt": idx+1,
    "from": c.get("from", {}).get("name", ""),
    "message": c.get("message", ""),
    "created_time": c.get("created_time", "")
} for idx, c in enumerate(comments)])

print(f"Tổng comment: {summary.get('total_count', len(comments))}")
print(df.head())

# Lưu CSV
df.to_csv("comments_page_ngonn.csv", index=False, encoding="utf-8-sig")
print("Đã lưu file comments_page_ngonn.csv")
