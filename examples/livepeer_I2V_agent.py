"""This script demonstrates how to create a Image-to-Video (I2V) agent using
Livepeer.

The implementation relies on the experimental `livepeer_ai` client SDK, which
can be found at: https://github.com/rickstaa/livepeer-ai-sdks/tree/main/sdks/python

For detailed instructions on how to use this SDK, please refer to the README.md
file in the repository: https://github.com/rickstaa/livepeer-ai-sdks/blob/main/sdks/python/README.md
"""  # noqa: E501

import argparse
import requests

import livepeer_ai
from livepeer_ai.rest import ApiException

from hive_agent import HiveAgent

configuration = livepeer_ai.Configuration(host="https://dream-gateway.livepeer.cloud")


class LivepeerI2VAgent:
    def __init__(self, configuration):
        self.configuration = configuration
        self.host = configuration.host

    def request_image_to_video(self, **kwargs):
        with livepeer_ai.ApiClient(self.configuration) as api_client:
            self.api_instance = livepeer_ai.DefaultApi(api_client)
            try:
                api_response = self.api_instance.image_to_video(**kwargs)
            except ApiException as e:
                raise Exception(e)
            return api_response

    def download_video(self, media):
        url = self.host + media.url
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open("I2V_output.mp4", "wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        out_file.write(chunk)
            print("Video downloaded successfully")
        else:
            print("Unable to download the video")

        return "I2V_output.mp4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Image-to-Video job using Livepeer AI subnet."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="The input image file path"
    )
    args = parser.parse_args()
    image_path = args.image

    # Retrieve the image from the file path
    with open(image_path, "rb") as f:
        image = f.read()

    LivepeerI2VAgent = LivepeerI2VAgent(configuration)

    livepeer_T2I_agent = HiveAgent(
        name="LivepeerI2VAgent",
        functions=[LivepeerI2VAgent.request_image_to_video],
        instruction="Convert text to image using the Livepeer AI service",
        port=8003,
    )

    # Check if the agent is setup correctly
    test_response = LivepeerI2VAgent.request_image_to_video(
        image=image,
        model_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        width=128,
        height=128,
    )

    # Download the first image from the response.
    video_path = LivepeerI2VAgent.download_video(test_response.images[0])
    print("Video saved at:", video_path)

    livepeer_T2I_agent.run()
