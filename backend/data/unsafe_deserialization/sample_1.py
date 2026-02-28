import pickle
data = b"\x80\x03c__builtin__\neval\nq\x00X\x0b\x00\x00\x00print('hi')q\x01\x85q\x02Rq\x03."
obj = pickle.loads(data)
