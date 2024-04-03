# This file is used to map deprecated setting names in a dictionary
# and print a message containing the old and the new names
#
# Examples:
# "my_dropdown": [                    | • changes in "my_dropdown"
#   ("removed_value", None, True)     |   • removal of value "removed_value"
#   ("choice1", "Choice One", False), |   • change of value from "choice1" to "Choice One"
# ],
# "use_lothar_of_hills":("character_choice", [    | • convert old dropdown/radio to new dropdown/radio:
#     ("Yes", "Lothar of the Hills"),             |   "use_lothar_of_hills" to "character_choice" |   • "Yes" becomes "Lothar of the Hills"
#     ("No", "Lotharica of the Plains")           |   • "No" becomes "Lotherica of the Plains"
# ]),                                             |   • original values removed
# "sweep_the_leg": [                  | • changes in "sweep_the_leg", convert checkbox to dropdown
#   (False, "No can defend.", False), |   • False becomes "No can defend."
#   (True, "Yes sensei!", False),     |   • True  becomes "Yes sensei!"
# ],                                  |   • original values not removed
#
# Possible values:
# None_|_string__|_tuple________|_list of tuples______________
# None | "value" | (v1,v2,bool) | [(v1,v2,bool), (v1,v2,bool)]
#
#     "input1": None,
#     "input1": "input1_name_change",
#     "input1": ("input2_transfer", [
#       ("value_from_input1", "value_for_input2", boolean (optional)),
#     ]),
#     "input1": [
#       ("value1", "new_value1", bool (optional)),
#       ("value2", "new value2", bool (optional)),
#     ],

deprecation_map = {
    "histogram_matching": None,
    "flip_2d_perspective": "enable_perspective_flip",
    "skip_video_for_run_all": "skip_video_creation",
    "color_coherence": [
        ("Match Frame 0 HSV", "HSV", False),
        ("Match Frame 0 LAB", "LAB", False),
        ("Match Frame 0 RGB", "RGB", False),
        ("color_coherence_source", [("Image", "Image Path", False), ("Video Input", "Video Init", False)], False),
        #, ("removed_value", None, True)              # for removed values, if we'll need in the future
    ],
    "color_coherence_video_every_N_frames": None,
    "legacy_colormatch": ("color_coherence_behavior", [("True", "Before"), ("False", "Before/After")]),
    "hybrid_composite": [
        (False, "None", False),
        (True, "Normal", False),
    ],
    "optical_flow_redo_generation": [
        (False, "None", False),
        (True, "DIS Fine", False),
    ],
    "optical_flow_cadence": [
        (False, "None", False),
        (True, "DIS Fine", False),
    ],
    "cn_1_resize_mode": [
        ("Envelope (Outer Fit)", "Outer Fit (Shrink to Fit)", False),
        ("Scale to Fit (Inner Fit)", "Inner Fit (Scale to Fit)", False),
    ],
    "cn_2_resize_mode": [
        ("Envelope (Outer Fit)", "Outer Fit (Shrink to Fit)", False),
        ("Scale to Fit (Inner Fit)", "Inner Fit (Scale to Fit)", False),
    ],
    "cn_3_resize_mode": [
        ("Envelope (Outer Fit)", "Outer Fit (Shrink to Fit)", False),
        ("Scale to Fit (Inner Fit)", "Inner Fit (Scale to Fit)", False),
    ],
    "use_zoe_depth": ("depth_algorithm", [("True", "Zoe+AdaBins (old)"), ("False", "Midas+AdaBins (old)")]),
}

def dynamic_num_to_schedule_formatter(old_value):
    return f"0:({old_value})"
    
for i in range(1, 6): # 5 CN models in total
    deprecation_map[f"cn_{i}_weight"] = dynamic_num_to_schedule_formatter
    deprecation_map[f"cn_{i}_guidance_start"] = dynamic_num_to_schedule_formatter
    deprecation_map[f"cn_{i}_guidance_end"] = dynamic_num_to_schedule_formatter

def handle_deprecated_settings(settings_json):
    for setting_name, deprecation_info in deprecation_map.items():
        if setting_name in settings_json:
            if deprecation_info is None:
                print(f"WARNING: Setting '{setting_name}' has been removed. It will be discarded and the default value used instead!")
            elif isinstance(deprecation_info, tuple):
                new_setting_name, value_map = deprecation_info
                old_value = str(settings_json.pop(setting_name))  # Convert the boolean value to a string for comparison
                new_value = next((v for k, v in value_map if k == old_value), None)
                if new_value is not None:
                    print(f"WARNING: Setting '{setting_name}' has been renamed to '{new_setting_name}' with value '{new_value}'. The saved settings file will reflect the change")
                    settings_json[new_setting_name] = new_value
            elif callable(deprecation_info):
                old_value = settings_json[setting_name]
                if isinstance(old_value, (int, float)):
                    new_value = deprecation_info(old_value)
                    print(f"WARNING: Value '{old_value}' for setting '{setting_name}' has been replaced with '{new_value}'. The saved settings file will reflect the change")
                    settings_json[setting_name] = new_value
            elif isinstance(deprecation_info, str):
                print(f"WARNING: Setting '{setting_name}' has been renamed to '{deprecation_info}'. The saved settings file will reflect the change")
                settings_json[deprecation_info] = settings_json.pop(setting_name)
            elif isinstance(deprecation_info, list):
                for old_value, new_value, is_removed in deprecation_info:
                    if settings_json[setting_name] == old_value:
                        if is_removed:
                            print(f"WARNING: Value '{old_value}' for setting '{setting_name}' has been removed. It will be discarded and the default value used instead!")
                        else:
                            print(f"WARNING: Value '{old_value}' for setting '{setting_name}' has been replaced with '{new_value}'. The saved settings file will reflect the change")
                            settings_json[setting_name] = new_value