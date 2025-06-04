set_names = [
    "train",
    "val",
    "test"
]

class_colors = {
    0: (0, 0, 0),  # Background - Black
    1: (255, 0, 0),  # Red
    2: (0, 255, 0),  # Green
    3: (0, 0, 255),  # Blue
    4: (255, 255, 0),  # Cyan
    5: (255, 0, 255),  # Magenta
    6: (0, 255, 255),  # Yellow
    7: (128, 0, 0),  # Maroon
    8: (0, 128, 0),  # Dark Green
    9: (0, 0, 128),  # Navy
    10: (128, 128, 0)  # Olive
}

datasets_classes = {
    "ppdd": {
        "background": 0,
        "transverse": 1,
        "longitudinal": 2,
        "slippage": 3,
        "crocodile": 4,
        "construction joint": 5,
        "rutting": 6
    },
    "rcd":{
        "background": 0,
        "crocodile": 1,
        "longitudinal": 2,
        "transverse": 3
    }
}