import os
import json
import pandas as pd
from convert_veracity_annotations import convert_annotations

def extract_tweet_info(file_path, event, thread_type, veracity, is_source):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            t = json.load(f)
            u = t.get('user', {})
            return {
                'event': event,
                'thread_type': thread_type,  # 'rumours' or 'non-rumours'
                'veracity': veracity,        # 'true', 'false', 'unverified', or 'non-rumour'
                'is_source': is_source,
                'tweet_id': t.get('id_str'),
                'user_id': u.get('id_str'),
                'user_screen_name': u.get('screen_name'),
                'text': t.get('text', ''),
                'parent_user_id': t.get('in_reply_to_user_id_str'),
                'followers': u.get('followers_count', 0),
                'verified_user': u.get('verified', False),
                'timestamp': t.get('created_at')
            }
    except:
        return None

def main():
    base_path = './pheme_data/all-rnr-annotated-threads'
    all_data = []

    if not os.path.exists(base_path):
        print("Error: Directory not found.")
        return

    for event in os.listdir(base_path):
        event_path = os.path.join(base_path, event)
        if not os.path.isdir(event_path) or event.startswith('.'): continue
        
        print(f"Flattening Event: {event}")

        for label in ['rumours', 'non-rumours']:
            label_path = os.path.join(event_path, label)
            if not os.path.isdir(label_path): continue

            for thread in os.listdir(label_path):
                thread_path = os.path.join(label_path, thread)
                if not os.path.isdir(thread_path) or thread.startswith('.'): continue
                
                # --- VERACITY LOGIC ---
                if label == 'non-rumours':
                    v_label = 'verified-news'
                else:
                    # Look for the veracity file
                    v_label = 'rumour-unlabeled' # Fallback
                    anno_path = os.path.join(thread_path, 'veracity-annotations.json')
                    if os.path.exists(anno_path):
                        with open(anno_path, 'r') as f:
                            try:
                                # Use your converter function
                                res = convert_annotations(json.load(f), string=True)
                                if res: v_label = res
                            except: pass

                # Process Tweets
                for sub in ['source-tweets', 'reactions']:
                    sub_path = os.path.join(thread_path, sub)
                    if os.path.exists(sub_path):
                        for f in os.listdir(sub_path):
                            if f.endswith('.json') and not f.startswith('.'):
                                tweet = extract_tweet_info(os.path.join(sub_path, f), event, label, v_label, (sub == 'source-tweets'))
                                if tweet: all_data.append(tweet)

    df = pd.DataFrame(all_data)
    # df.to_csv('pheme_master.csv', index=False)
    df.to_csv('pheme_master_v2.csv', index=False)
    print(f"Done! Created 'pheme_master.csv' with {len(df)} rows.")

if __name__ == "__main__":
    main()