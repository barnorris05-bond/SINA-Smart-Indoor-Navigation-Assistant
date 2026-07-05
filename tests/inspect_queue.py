import depthai as dai
import inspect

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    output = cam.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.BGR888p,
        fps=30
    )

    print("Method:")
    print(output.createOutputQueue)

    print("\nSignature:")

    try:
        print(inspect.signature(output.createOutputQueue))
    except Exception as e:
        print(e)

    print("\nHelp:")
    help(output.createOutputQueue)