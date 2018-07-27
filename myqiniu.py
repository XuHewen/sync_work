from qiniu import Auth
from qiniu import BucketManager, build_batch_delete
access_key = 'U8VGSrOjjPpoPFEG-qHUJrywNL743V2PsrMjW6WM'
secret_key = 'ZcWIywOvp1NaaUNMU0o3ml8OTC-kk8TG_5e_T0kN'
BUCKET_NAME='img4wc-dev'

q = Auth(access_key, secret_key)

bucket = BucketManager(q)
prefix = 'thumb/'

ret, eof, info = bucket.list(BUCKET_NAME, prefix, limit=2000)

keys = []
for x in ret['items']:
    print(x['key'])
    dkeys.append(x['key'])

print(len(keys))

# ops = build_batch_delete(BUCKET_NAME, keys)
# ret, info = bucket.batch(ops)
# print(info)