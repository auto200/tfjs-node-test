import { load as loadPosenet } from "@tensorflow-models/posenet";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { createCanvas, loadImage } from "canvas";

const WIDTH = 580;
const HEIGHT = 435;

const canvas = createCanvas(WIDTH, HEIGHT);
const ctx = canvas.getContext("2d");

(async () => {
  const image = await loadImage("./test-photo.jpg");
  ctx.drawImage(image, 0, 0);
  const imageData = tf.browser.fromPixels(canvas as any);
  const poseNet = await loadPosenet({
    architecture: "ResNet50",
    inputResolution: { width: WIDTH, height: HEIGHT },
    outputStride: 16,
  });
  for (let i = 0; i < 100; i++) {
    const startTime = performance.now();
    const pose = await poseNet.estimateSinglePose(imageData);
    console.log(performance.now() - startTime);
  }
})();
