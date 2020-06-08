import inputs
import model_tools as mt
import grayscale_model as gm
import colour_model as cm
import vgg19_model as vgg

configs = inputs.config()

# Input from file & Data Augmentation
in_conf = configs["inputs"]
if (in_conf["all"]):
    inputs.gdrive()
    inputs.serialize()
    inputs.augment()
    inputs.grayscale()
else:
    if (in_conf["download"]):
        inputs.gdrive()
    if (in_conf["serialize"]):
        inputs.serialize()
    if (in_conf["augment"]):
        inputs.augment()
    if (in_conf["grayscale"]):
        inputs.grayscale()

# Neural Network for Colour Images 150x150
gm_conf = configs["grayscale_model"]
if (gm_conf["load"]):
    grayscale_model = mt.load_model("baseline_model")
else:
    grayscale_model = gm.get_model()
if (gm_conf["summary"]):
    grayscale_model.summary()
if (gm_conf["run"]):
    gm.run(grayscale_model,
           plot=(gm_conf["plot"] and configs["matplotlib_gui"]),
           test=gm_conf["test"],
           save=gm_conf["save"])

# Neural Network for Colour Images 150x150

cm_conf = configs["colour_model"]
if (cm_conf["load"]):
    colour_model = mt.load_model("colour_model")
else:
    colour_model = cm.get_model()
if (cm_conf["summary"]):
    colour_model.summary()
if (cm_conf["run"]):
    cm.run(colour_model,
           mix=cm_conf["mix"],
           plot=(cm_conf["plot"] and configs["matplotlib_gui"]),
           test=cm_conf["test"],
           save=cm_conf["save"])

# VGG19 Transfer Network for Colour Images 224x224

vgg_conf = configs["vgg19_model"]
if (vgg_conf["load"]):
    vgg_model = mt.load_model("topVGG19model")
else:
    vgg_model = vgg.get_model()
if (vgg_conf["convolve"]):
    vgg.imagenet()
    vgg.vgg_conv()
if (vgg_conf["summary"]):
    vgg_model.summary()
if (vgg_conf["run"]):
    vgg.run(vgg_model,
            plot=(vgg_conf["plot"] and configs["matplotlib_gui"]),
            test=vgg_conf["test"],
            save=vgg_conf["save"],
            conf_matrix=(vgg_conf["confusion"] and configs["matplotlib_gui"]))

# Large-Scale Predicting
if (configs["predict"]["run"]):
    for filename in configs["predict"]["filenames"]:
        vgg.grid(filename, plot=(configs["predict"]
                                 ["grid"] and configs["matplotlib_gui"]))
