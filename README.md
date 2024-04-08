
# <div style="text-shadow:red 1px 1px 10px;"><b><span style='color:red;opacity:0.5;font-size:1.25em;'>D</span><span style="position:absolute;left:5px;top:7px;color:gold;z-index:10;">R</span>eforum</b> - <em style="font-size:0.75em;"><span style="color:gold;">r</span>eallybigname deforum <span style="font-size:0.6em;">(experimental branch)</span></em></div>

#### I started this branch in like June of 2023 and it became very different over time. I just finally released it to the public and it's now April 2024. See, I used it as my experimental playground, and didn't limit myself as I would have with the main branch. I wanted to change some things fundamentally and experiment. There's a lot on the back end that are invisible, but I'll try to start documenting the major front-end changes and features here:

- **Cadence**
  - **Cadence easing schedule** - allows more slideshow-like cadence runs, and other timing effects.
  - **Cadence flow easing schedule** - changes easing of the flow within the eased cadence frames. You can mix and match the cadence easing and cadence flow easing schedules for different effects, although I usually align them.
  - **Cadence flow warp factor schedule** - cadence flow has been further enhanced, and also now has a warp factor schedule which determines the amount that the next image is warped towards the previous image at the start of cadence. Warp factors may even be mixed, with weighting.
  - **Cadence behavior now always dual image** - It no longer starts with the same image it finishes with on the first cycle of cadence. I store frames in a history until two are available, and made it so that cadence is always provided with two images.
  - **Cadence fundamentally changed** - Old cadence all happened in one loop. My cadence routines each loop over a cadence object which stores all prev/next 'turbo' frames. Each function does what it needs and passes it on, and only at the very end of the cadence cycle does it mix the prev and next images for the final result. This can be slower at times, but also offers a much smoother cadence flow as well as allowing for various processes during cadence without them all interfering with each other.  
- **Temporal flow** is a new feature which gets the flow from a previous frame to the current frame and applies the flow to the current frame with a flow factor schedule. You can change the frame it targets with a schedule as well. And, it has some optional behaviors inside of cadence frames: Forward or Bounce.
- **Morphological transformations** are a new feature using cv2.morphologyEx function. I get the flow from the current frame to a temporary morph'd frame, and apply that flow with a flow factor schedule. With 7 possible operations using 11 possible kernels, there are a lot of combinations, and there is an iteration schedule for the function. It also has modes that control how the function operates behind the scenes, as grayscale, 3-color, or just a bitmap. Bitmap has further low/high cutoff control schedules. I also added a direct mode that doesn't use flow, and simply returns the direct result of the transform, even if it's in grayscale or bitmap. This was for testing originally, but you can do some cool things with that as well.
- **Color coherence enhancements**:
  - **Separated Color coherence type and Color coherence source in UI**. *(Video color matching was previously always done with LAB behind the scenes. Now any color match can be used with any source)*
  - **Color coherence sources** can be *First Frame, Image Path, Video Init, or Video Path*
    - *Video path* is a custom path for a color match video only (can be different than video init)
    - compact textbox designation for "<b>from|to|increment</b>" (ie. "<b>0|-1|1</b>")
  - **Added HahneC color-matcher** types to color coherence, some are reliable even if only applied After generation
  - **Added color coherence alpha schedule** - Allows you to alter the amount of color coherence to the selected source. You can make it 0.01 so it slowly picks up the colors, or you can make it 0.9 if you want to give the color pallette just a little bit of wiggle room. Behind the scenes it mixes the current image with the color coherence sample and uses the mix for that frame. But, to prevent the actual colors from being blended, I mix the two images with a special per-pixel function, swapping rather than blending any. 
- **Loop compositing** - All new feature which changes a fundamental part of how Deforum can work.   
  - **Mix prev_imgs** - Normally, the current image becomes the previous image on the next cycle. Instead of just doing the swap, I add the ability to mix the old previous image with the new previous image, with an alpha schedule, as well as options to use a bunch of layer compositing functions like multiply, luma darken only, screen, difference, just to name a few. And, you can flip flop the order of the two images for the composite.
  - **Loop conform flow alignment schedule** - Unless you want blurriness, it's no good to composite two images unless they are aligned, right? Well, that's where the optional '**conform**' comes in. It allows you to attempt to make the image conform to the new image's shape before compositing. using optical flow, and optional iterations. But, it doesn't just try to align the old image with the new. Valid schedule values are between 0 and 1 with 0 being the old and 1 being the new. For instance:
    - At 0, it tries to align the new frame with the old frame's shape.
    - At 1, it tries to align the old frame with the new frame's shape.
    - At 0.5, it tries to aligns both images half-way towards each other's shape.
- **Hybrid compositing** - Added the same layer compositing functions seen in 'Loop compositing' above as options for hybrid compositing. So, the conform function allows you to do a lot of video motion stuff without using the hybrid motion section at all, unless you want to. Rather than copying the motion from two frames in the video and applying the motion to the rendering, this just tries to force the current frame to conform with the current frame of video. Simple, yet very powerful. Amazingly, I suggest 1 iteration and Farneback flow for consistency. Also, you can use it during cadence, even some long cadence can hold together with fast motion. 
- **Hybrid motion:**
  - Added option **After Generation** - can be very useful in certain situations where it's paying more attention to ControlNet weights than strength and init image. 
  - Added **new experimental Matrix Flow** - Still working on it, but the idea with Matrix Flow is that it doesn't just use a matrix for camera motion and it doesn't just use optical flow for motion in the scene, and instead uses both. However, unlike my other Perspective and Affine RANSAC camera tracking modes, Matrix Flow actually hooks right into the same functions that make the 2D or 3D animation keyframing work. I also use optical flow, cancelling out the other motion from the matrix transformation. The promise of Matrix Flow if it worked nicely is that 3D objects from a video could also take advantage of the depth routines of 3D animation keyframing, aligning the animation maths with the world in the video. But, there are challenges in this approach, namely that there is no reliable way to estimate z_translation and fov nicely from a video source. But, I do have it partially working. More experimentation needed. I can't guarantee it works with all other features, as it is experimental.
- **Fixed Resume from timestring**. Can delete any number of frames going backwards and resume. Works flawlessly on everything except certain temporal flow cases, which I still need to account for.
- **Added fancy icons** next to all section titles
- **Customized CSS** - converted to less.css behind the scenes
- **Optional CSS Button** located on the Hybrid Video tab enables some different fonts, styles, and animations, *including tabs that spin and take off like helicopters if you hover for a bit*... Or the rocket icon 🚀 for the Motion tab that *actually takes off then comes back and lands*. Or, *active tabs that periodically glow a little*.

- ***<span style="color:yellow;text-shadow:black -1px -1px 2px">I'm sure there's more new stuff, but I have to stop typing for now... The truth is, there are a ton of experimental features still in the code, just disabled. This is all just what I have kept, haha.***

### Be sure to check '**Show more info**' (top of UI) for help
- **Expandable help is also provided for some features**. 

### Below is the original content before my branch:
<hr />

# Deforum Stable Diffusion — official extension for AUTOMATIC1111's webui

<p align="left">
    <a href="https://github.com/deforum-art/sd-webui-deforum/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/sd-webui-deforum/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/sd-webui-deforum/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/deforum-art/deforum-for-automatic1111-webui"></a>
    <a href="https://github.com/deforum-art/sd-webui-deforum/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/deforum-art/deforum-for-automatic1111-webui"></a>
    </a>
</p>

## Need help? See our [FAQ](https://github.com/deforum-art/sd-webui-deforum/wiki/FAQ-&-Troubleshooting)

## Getting Started

1. Install [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/).

2. Now two ways: either clone the repo into the `extensions` directory via git commandline launched within in the `stable-diffusion-webui` folder

```sh
git clone https://github.com/deforum-art/sd-webui-deforum extensions/deforum
```

Or download this repository, locate the `extensions` folder within your WebUI installation, create a folder named `deforum` and put the contents of the downloaded directory inside of it. Then restart WebUI.

3. Open the webui, find the Deforum tab at the top of the page.

4. Enter the animation settings. Refer to [this general guide](https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit) and [this guide to math keyframing functions in Deforum](https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing). However, **in this version prompt weights less than zero don't just like in original Deforum!** Split the positive and the negative prompt in the json section using --neg argument like this "apple:\`where(cos(t)>=0, cos(t), 0)\`, snow --neg strawberry:\`where(cos(t)<0, -cos(t), 0)\`"

5. To view animation frames as they're being made, without waiting for the completion of an animation, go to the 'Settings' tab and set the value of this toolbar **above zero**. Warning: it may slow down the generation process.

![adsdasunknown](https://user-images.githubusercontent.com/14872007/196064311-1b79866a-e55b-438a-84a7-004ff30829ad.png)


6. Run the script and see if you got it working or even got something. **In 3D mode a large delay is expected at first** as the script loads the depth models. In the end, using the default settings the whole thing should consume 6.4 GBs of VRAM at 3D mode peaks and no more than 3.8 GB VRAM in 3D mode if you launch the webui with the '--lowvram' command line argument.

7. After the generation process is completed, click the button with the self-describing name to show the video or gif result right in the GUI!

8. Join our Discord where you can post generated stuff, ask questions and more: https://discord.gg/deforum. <br>
* There's also the 'Issues' tab in the repo, for well... reporting issues ;) 

9. Profit!

## Known issues

* This port is not fully backward-compatible with the notebook and the local version both due to the changes in how AUTOMATIC1111's webui handles Stable Diffusion models and the changes in this script to get it to work in the new environment. *Expect* that you may not get exactly the same result or that the thing may break down because of the older settings.

## Screenshots

Amazing raw Deforum animation by [Pxl.Pshr](https://www.instagram.com/pxl.pshr):
* Turn Audio ON!

(Audio credits: SKRILLEX, FRED AGAIN & FLOWDAN - RUMBLE (PHACE'S DNB FLIP))

https://user-images.githubusercontent.com/121192995/224450647-39529b28-be04-4871-bb7a-faf7afda2ef2.mp4

Setting file of that video: [here](https://github.com/deforum-art/sd-webui-deforum/files/11353167/PxlPshrWinningAnimationSettings.txt).

<br>

Main extension tab:

![image](https://user-images.githubusercontent.com/121192995/226101131-43bf594a-3152-45dd-a5d1-2538d0bc221d.png)

Keyframes tab:

![image](https://user-images.githubusercontent.com/121192995/226101140-bfe6cce7-9b78-4a1d-be9a-43e1fc78239e.png)
