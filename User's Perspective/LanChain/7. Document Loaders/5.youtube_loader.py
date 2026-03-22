# --------- This is to get video meta data ----------
# from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
# # Basic transcript loading
# loader = YoutubeLoaderDL.from_youtube_url(
#     "https://www.youtube.com/watch?v=dQw4w9WgXcQ", add_video_info=True
# )
# documents = loader.load()
# documents[0].metadata


from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=nhLZKsxEwxM", add_video_info=False
)

transcript = loader.load()

print(transcript[0].page_content[:200])
