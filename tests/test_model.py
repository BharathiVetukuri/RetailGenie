import pytest
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def test_model_loading():
    """Test if the model can be loaded correctly"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        assert model is not None
        assert tokenizer is not None
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")


def test_model_inference():
    """Test if the model can perform inference"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

        # Test input
        input_text = "translate English to German: Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate output
        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert isinstance(decoded_output, str)
        assert len(decoded_output) > 0
    except Exception as e:
        pytest.fail(f"Model inference failed: {str(e)}")


def test_model_output_format():
    """Test if the model output is in the correct format"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

        # Test input
        input_text = "translate English to German: Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate output
        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check output format
        assert isinstance(decoded_output, str)
        assert len(decoded_output.split()) > 0
    except Exception as e:
        pytest.fail(f"Model output format test failed: {str(e)}")
