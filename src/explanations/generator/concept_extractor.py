from typing import List, Optional, Union
from pydantic import BaseModel, Field

import os
import logging
import re
import base64
import io
import torch
from difflib import SequenceMatcher
from PIL import Image
import numpy as np

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.prompts.prompts import NLEPrompts

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptExtractionResult(BaseModel):
    """Pydantic model to store concept extraction results"""

    class_name: str = Field(..., description="Name of the class")
    description: str = Field(..., description="Generated description")
    concepts: List[str] = Field(
        default_factory=list, description="List of extracted concepts"
    )
    raw_response: str = Field("", description="Raw response from the API")

    def model_post_init(self, __context) -> None:
        """Post-initialization processing"""
        if not self.raw_response and self.description:
            self.raw_response = (
                f"Description: {self.description}\nConcepts: {'; '.join(self.concepts)}"
            )


class ConceptExtractor(BaseModel):
    """
    Pydantic model for generating descriptions from class names using GPT
    and extracting concepts from those descriptions using LangChain
    """

    api_key: Optional[str] = Field(None, description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="GPT model to use")
    vlm_model: str = Field(default="gpt-4o", description="Vision-Language model to use")
    llm: Optional[ChatOpenAI] = Field(
        None, description="LangChain ChatOpenAI instance for text-only tasks"
    )
    vlm: Optional[Union[ChatOpenAI, object]] = Field(
        None, description="VLM instance (ChatOpenAI or InstructBLIP)"
    )
    vlm_processor: Optional[object] = Field(None, description="InstructBLIP processor")

    # Generation parameters
    max_total_descriptions: int = Field(
        default=100, description="Maximum total descriptions to generate"
    )
    similarity_threshold: float = Field(
        default=0.8, description="Similarity threshold for duplicate removal"
    )

    # PromptTemplate attributes
    description_template_llm: Optional[Union[PromptTemplate, str]] = Field(
        None, description="Template for description generation"
    )
    description_template_vlm: Optional[Union[PromptTemplate, str]] = Field(
        None, description="Template for description generation"
    )

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context) -> None:
        """Post-initialization processing"""

        """Initialize API key from environment if not provided"""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set api_key parameter or OPENAI_API_KEY environment variable."
            )

        """Initialize LangChain ChatOpenAI models for both text-only and vision-language tasks"""
        # Initialize text-only model
        self.llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_key=self.api_key,
            temperature=1.0,
            max_tokens=4096,
        )

        # Initialize vision-language model
        if self.vlm_model == "gpt-4o":
            self.vlm = ChatOpenAI(
                model=self.vlm_model,
                openai_api_key=self.api_key,
                temperature=0.7,
                max_tokens=4096,
            )
        elif self.vlm_model == "Qwen/Qwen2.5-VL-7B-Instruct":
            from transformers import (
                AutoProcessor,
                AutoModelForVision2Seq,
                BitsAndBytesConfig,
            )
            import torch

            print(f"Loading {self.vlm_model} with 8bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.vlm_processor = AutoProcessor.from_pretrained(
                self.vlm_model,
                local_files_only=False,
            )
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                self.vlm_model,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=False,
            )
            print("✓ Qwen2.5-VL model loaded with 8bit quantization")

        """Initialize prompt templates"""
        # Use visual concepts prompt as default for text-only tasks
        self.description_template_llm = NLEPrompts.get_llm_visual_concepts_prompt()
        self.description_template_vlm = NLEPrompts.get_vlm_visual_concepts_prompt()

    def generate_concepts_from_llm(
        self,
        class_name: str,
        num_concepts: Optional[int] = None,
    ) -> List[str]:
        if num_concepts is None:
            num_concepts = self.max_total_descriptions

        print(f"Processing class '{class_name}'")

        all_concepts = []
        while len(all_concepts) < num_concepts:
            # Prepare existing concepts text to avoid repetition
            existing_concepts_text = self._prepare_existing_concepts_text(all_concepts)

            # Generate multiple concepts with existing concepts context
            formatted_prompt = self.description_template_llm.format(
                class_name=class_name, existing_concepts=existing_concepts_text
            )

            _response = self.llm.invoke(formatted_prompt)
            _concepts = self._parse_descriptions(_response.content.strip())
            all_concepts.extend(_concepts)
            all_concepts = self._remove_duplicate_concepts(all_concepts)
            print(f"Generated {len(all_concepts)} concepts")

        all_concepts = all_concepts[:num_concepts]

        return all_concepts

    def generate_concepts_from_vlm(
        self,
        image: torch.Tensor,
        class_name: str,
        num_concepts: Optional[int] = None,
    ) -> List[str]:
        if num_concepts is None:
            num_concepts = self.max_total_descriptions

        print(f"Processing class '{class_name}'")

        all_concepts = []
        while len(all_concepts) < num_concepts:
            # Prepare existing concepts text to avoid repetition
            existing_concepts_text = self._prepare_existing_concepts_text(all_concepts)

            # Generate multiple concepts with existing concepts context
            # Use the VLM prompt template from prompts.py
            formatted_prompt = self.description_template_vlm.format(
                class_name=class_name, existing_concepts=existing_concepts_text
            )

            # Generate response based on VLM type
            response_content = ""
            try:
                if self.vlm_model == "gpt-4o":
                    # For OpenAI VLM, we need to send the image along with the prompt
                    image_base64 = self._tensor_to_base64(image)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": formatted_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ]
                    _response = self.vlm.invoke(messages)
                    response_content = _response.content.strip()
                elif self.vlm_model == "Qwen/Qwen2.5-VL-7B-Instruct":
                    response_content = self._generate_with_qwen_vlm(
                        image, formatted_prompt
                    )
                else:
                    print(
                        f"Warning: Unsupported VLM model '{self.vlm_model}'. Skipping concept generation."
                    )
                    continue
            except Exception as e:
                print(f"Error generating concepts with {self.vlm_model}: {e}")
                continue

            _concepts = self._parse_descriptions(response_content)
            all_concepts.extend(_concepts)
            all_concepts = self._remove_duplicate_concepts(all_concepts)
            print(f"Generated {len(all_concepts)} concepts")

        all_concepts = all_concepts[:num_concepts]

        return all_concepts

    def _generate_with_qwen_vlm(self, image: torch.Tensor, prompt: str) -> str:
        """Generate response using Qwen2.5-VL model"""

        # Convert tensor to PIL Image
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 3:  # CHW to HWC
            image = image.permute(1, 2, 0)

        # Convert to numpy and scale to 0-255
        image_np = image.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        pil_image = Image.fromarray(image_np)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process with Qwen
        inputs = self.vlm_processor(text=text, images=pil_image, return_tensors="pt")

        # Move to device
        device = next(self.vlm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.vlm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=0.8,
                pad_token_id=self.vlm_processor.tokenizer.eos_token_id,
            )

        # Decode response
        generated_text = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)

        # Extract response content
        if "assistant" in generated_text:
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text

        return response

    def _prepare_existing_concepts_text(self, existing_concepts: List[str]) -> str:
        """Prepare text for existing concepts to avoid repetition in prompt"""
        if not existing_concepts:
            return ""

        existing_text = "Already generated concepts (DO NOT repeat these):\n"
        for concept in existing_concepts:
            existing_text += f"- {concept}\n"
        existing_text += "\nGenerate NEW and DIFFERENT visual features:\n\n"

        return existing_text

    def _remove_duplicate_concepts(self, concepts: List[str]) -> List[str]:
        """Remove duplicate and similar concepts"""
        if not concepts:
            return []

        unique_concepts = []
        seen_concepts = set()

        for concept in concepts:
            concept_clean = concept.strip().lower()

            # Skip if exact duplicate (case-insensitive)
            if concept_clean in seen_concepts:
                continue

            # Check for similar concepts
            is_similar = False
            for unique_concept in unique_concepts:
                if (
                    self._calculate_similarity(concept, unique_concept)
                    > self.similarity_threshold
                ):
                    logger.debug(
                        f"Filtered similar concept: '{concept}' (similar to '{unique_concept}')"
                    )
                    is_similar = True
                    break

            if not is_similar:
                unique_concepts.append(concept)
                seen_concepts.add(concept_clean)

        return unique_concepts

    def _parse_descriptions(self, response: str) -> List[str]:
        """Parse descriptions from LLM response - extract 1-3 word concepts only"""
        descriptions = []
        text = response.strip()

        # Split by common separators and newlines
        parts = re.split(r"[;\n,]|\s+and\s+|\s+or\s+|\s+-\s+", text)

        for part in parts:
            # Clean up the concept
            concept = part.strip().lower()
            concept = re.sub(r"^(a|an|the)\s+", "", concept)  # Remove articles
            concept = re.sub(r"^\d+\.\s*", "", concept)  # Remove numbering
            concept = re.sub(r"^-\s*", "", concept)  # Remove dashes
            concept = re.sub(r"\s+", " ", concept)  # Normalize whitespace
            concept = concept.strip("\"'")  # Remove quotes

            # Only keep concepts with 1-3 words
            words = concept.split()
            if 1 <= len(words) <= 3 and len(concept) > 0:
                descriptions.append(concept)

        return self._remove_duplicates(descriptions)

    def _remove_duplicates(self, descriptions: List[str]) -> List[str]:
        """Remove duplicates while preserving order and ensure 1-3 words"""
        seen = set()
        unique_descriptions = []
        for desc in descriptions:
            word_count = len(desc.split())
            if desc not in seen and len(desc) > 2 and 1 <= word_count <= 3:
                seen.add(desc)
                unique_descriptions.append(desc)
        return unique_descriptions

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher"""
        return SequenceMatcher(
            None, text1.lower().strip(), text2.lower().strip()
        ).ratio()

    def _tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """Convert tensor to base64 string for VLM API"""
        # Convert tensor to PIL Image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present

        # Ensure tensor is in [0, 1] range
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        # Convert to PIL Image
        if tensor.shape[0] == 3:  # CHW format
            tensor = tensor.permute(1, 2, 0)  # Convert to HWC

        # Convert to numpy and ensure uint8
        tensor_np = (tensor.cpu().numpy() * 255).astype(np.uint8)

        # Convert to PIL Image and then to base64
        image = Image.fromarray(tensor_np)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")
