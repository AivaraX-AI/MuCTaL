import sys
a=sys.argv[1]
b=bool(int(sys.argv[2]))
print(a)
print(b)
if b:
    print(b,"is true")
else:
    print(b,"is false")