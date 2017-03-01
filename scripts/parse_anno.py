import json

# download the raw annotations from hypothes
# for example https://hypothes.is/api/search?_separate_replies=true&group=__world__&limit=200&offset=0&order=asc&sort=created&uri=https%3A%2F%2Fexacity.github.io%2Fdeeplearningbook-chinese%2FChapter2_linear_algebra%2F
anno = open('raw.txt','r').read()
json_anno = json.loads(anno)

rows = json_anno['rows']
replies = json_anno['replies']

cleaned_keys = ['user', 'text']
cleaned_replies = []
cleaned_rows = []

for reply in replies:
    cleaned_reply = {k:reply[k] for k in ['user', 'text']}
    cleaned_reply['time'] = reply['updated'][:16]
    cleaned_replies.append(str(cleaned_reply))

for row in rows:
    cleaned_row = {k:row[k] for k in ['user', 'text']}
    origin_text = row['target'][0]['selector'][3]
    cleaned_row['origin_text'] = (origin_text['prefix'] + "  !!!" + origin_text['exact'] + "!!!  " + origin_text['suffix']).replace('\n', '')
    cleaned_row['time'] = row['updated'][:16]
    cleaned_rows.append(str(cleaned_row))

# write to new file
f = open('annotations.txt', 'w')
f.write('\n'.join(cleaned_rows))
f.write('\n\n=============================   Replies   =============================\n\n')
f.write('\n'.join(cleaned_replies))
f.close()
