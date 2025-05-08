from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
    SYSTEM_PROMPT = """
    # Rol y Propósito
    Sos un asistente de inteligencia artificial para un sistema de preguntas frecuentes (FAQ) de comercio electrónico. Tu tarea es sintetizar una respuesta coherente y útil basada en la pregunta dada y el contexto relevante recuperado de una base de conocimiento.
    
    # Pautas:
    1. Proporcioná una respuesta clara y concisa a la pregunta.
    2. Usá únicamente la información proveniente del contexto relevante para respaldar tu respuesta.
    3. El contexto se recupera en función de la similitud coseno, por lo que puede que falte información o que haya contenido irrelevante.
    4. Sé transparente cuando no haya suficiente información para responder completamente la pregunta.
    5. No inventes ni infieras información que no esté presente en el contexto proporcionado.
    6. Si no podés responder la pregunta con base en el contexto dado, indicálo claramente.
    7. Mantené un tono útil y profesional, apropiado para atención al cliente.
    8. Apegate estrictamente a las políticas y directrices de la empresa utilizando únicamente la base de conocimiento proporcionada.

    Revisá la pregunta del usuario:
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content", "category"]
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Retrieved information:\n{context_str}",
            },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)
