import depthai as dai

print("=" * 50)
print("DepthAI Version:", dai.__version__)
print("=" * 50)

print("\nPipeline Class")
print(dai.Pipeline)

print("\nAvailable Nodes")

for node in dir(dai.node):
    if not node.startswith("_"):
        print(node)