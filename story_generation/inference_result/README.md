## Data Format of the Inferenced Results

1. Magic Search:
The inferenced file from magic search is a list of dictionary, where the data format of each dictionary is:
```yaml
{  
   "prefix_text": The story title.
   "reference_continuation_text": The human-written story.
   "generated_result": {
        "0": The generated result based on the top-1 retrieved image from the image index
            {
              "image_name": The file name of the retrieved image.
              "generated_continuation_text": The generated result from magic search.
              "generated_full_text": The story title + the generated result
            },
        ...
   }
}
```
