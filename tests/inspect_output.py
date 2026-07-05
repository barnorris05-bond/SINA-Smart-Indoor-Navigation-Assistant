import depthai as dai

with dai.Pipeline() as pipeline:

    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    rgb = cam.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.BGR888p,
        fps=30
    )

    print(type(rgb))
    print(dir(rgb))