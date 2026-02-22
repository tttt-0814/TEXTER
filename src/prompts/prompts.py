from langchain.prompts import PromptTemplate


class NLEPrompts:
    @staticmethod
    def get_llm_visual_concepts_prompt() -> PromptTemplate:
        """Get prompt template for generating simple visual concepts including object and context"""
        return PromptTemplate(
            input_variables=["class_name", "existing_concepts"],
            template="""Important guidelines for generating visual concepts:
1. Generate GENERAL concepts that can apply to many different photos of the same object type
2. Include both OBJECT features (e.g., shape, color, parts) AND CONTEXT features (e.g., background, environment, setting)
3. Focus on features that are commonly visible and distinguishable
4. Keep concepts short and specific (1-3 words)
5. DO NOT include class names or object names directly (e.g., avoid "shark cage" for shark images, use "metal bars" instead)

Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- long tail
- large eyes
- gray fur
- trees
- branches
- forest

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- black screen
- rectangular shape
- remote control
- living room
- wall
- stand

Q: What are useful visual features for distinguishing a koi fish in a photo?
A: There are several useful visual features to tell there is a koi fish in a photo:
- orange color
- fish body
- fins
- water
- pond
- ripples

Q: What are useful features for distinguishing a {class_name} in a photo?
{existing_concepts}A: There are several useful visual features to tell there is a {class_name} in a photo. Generate approximately 10 visual concepts to provide comprehensive coverage:
-""",
        )

    @staticmethod
    def get_vlm_visual_concepts_prompt() -> str:
        """Get VLM prompt text for generating visual concepts from images"""
        return """Important guidelines for generating visual concepts:
1. Generate DETAILED and SPECIFIC concepts that can apply to this image.
2. Include both OBJECT features (e.g., shape, color, parts) AND CONTEXT features (e.g., background, environment, setting)
3. Focus on features that are commonly visible and distinguishable
4. Keep concepts short and specific (1-3 words)
5. DO NOT include class names or object names directly (e.g., avoid "shark cage" for shark images, use "metal bars" instead)

Examples:
Q: Look at this image carefully. Based on what you can actually see in the image, identify useful visual features that help distinguish this as a koi fish.
A: There are several useful visual features to tell there is a koi fish in a photo:
- bright orange scales
- curved tail fin
- spotted pattern
- long body
- pointed snout
- water surface


Q:Look at this image carefully. Based on what you can actually see in the image, identify useful visual features that help distinguish this as a {class_name}.
{existing_concepts}A: There are several useful visual features to tell there is a {class_name} in a photo. Generate approximately 10 visual concepts to provide comprehensive coverage:
- """
