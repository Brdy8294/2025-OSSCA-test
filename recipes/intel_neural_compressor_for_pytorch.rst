Ease-of-use quantization for PyTorch with Intel® Neural Compressor
==================================================================

Overview
--------

대부분의 딥러닝 애플리케이션은 추론 시 32비트 부동소수점 정밀도를 사용합니다. 그러나 FP8과 같은 저정밀 데이터 타입은 뛰어난 성능 향상 효과로 인해 점점 더 주목받고 있습니다. 저정밀 방식을 채택할 때의 핵심 과제는 미리 정의된 요구사항을 충족하면서 정확도 손실을 최소화하는 것입니다.
Intel® Neural Compressor는 정확도 기반 자동 튜닝 전략을 통해 PyTorch를 확장하여, 사용자가 인텔 하드웨어에서 최적의 양자화 모델을 빠르게 찾을 수 있도록 위 문제를 해결하는 것을 목표로 합니다.
Intel® Neural Compressor는 `Github <https://github.com/intel/neural-compressor>`_ 에서 확인할 수 있는 오픈소스 프로젝트입니다.

Features
--------

- **Ease-of-use API:** Intel® Neural Compressor is re-using the PyTorch ``prepare``, ``convert`` API for user usage.

- **Accuracy-driven Tuning:** Intel® Neural Compressor supports accuracy-driven automatic tuning process, provides ``autotune`` API for user usage.

- **Kinds of Quantization:** Intel® Neural Compressor supports a variety of quantization methods, including classic INT8 quantization, weight-only quantization and the popular FP8 quantization. Neural compressor also provides the latest research in simulation work, such as MX data type emulation quantization. For more details, please refer to `Supported Matrix <https://github.com/intel/neural-compressor/blob/master/docs/source/3x/PyTorch.md#supported-matrix>`_.

Getting Started
---------------

Installation
~~~~~~~~~~~~

.. code:: bash

    # install stable version from pip
    pip install neural-compressor-pt
..

**Note**: Neural Compressor provides automatic accelerator detection, including HPU, Intel GPU, CUDA, and CPU. To specify the target device, ``INC_TARGET_DEVICE`` is suggested, e.g., ``export INC_TARGET_DEVICE=cpu``.


Examples
~~~~~~~~~~~~

This section shows examples of kinds of quantization with Intel® Neural compressor

FP8 Quantization
^^^^^^^^^^^^^^^^

**FP8 Quantization** is supported by Intel® Gaudi®2&3 AI Accelerator (HPU). To prepare the environment, please refer to `Intel® Gaudi® Documentation <https://docs.habana.ai/en/latest/index.html>`_.

Run the example,

.. code-block:: python

    # FP8 Quantization Example
    from neural_compressor.torch.quantization import (
        FP8Config,
        prepare,
        convert,
    )

    import torch
    import torchvision.models as models

    # Load a pre-trained ResNet18 model
    model = models.resnet18()

    # Configure FP8 quantization
    qconfig = FP8Config(fp8_config="E4M3")
    model = prepare(model, qconfig)

    # Perform calibration (replace with actual calibration data)
    calibration_data = torch.randn(1, 3, 224, 224).to("hpu")
    model(calibration_data)

    # Convert the model to FP8
    model = convert(model)

    # Perform inference
    input_data = torch.randn(1, 3, 224, 224).to("hpu")
    output = model(input_data).to("cpu")
    print(output)

..

Weight-only Quantization
^^^^^^^^^^^^^^^^^^^^^^^^

**Weight-only Quantization** is also supported on Intel® Gaudi®2&3 AI Accelerator. The quantized model could be loaded as below.

.. code-block:: python

    from neural_compressor.torch.quantization import load

    # The model name comes from HuggingFace Model Hub.
    model_name = "TheBloke/Llama-2-7B-GPTQ"
    model = load(
        model_name_or_path=model_name,
        format="huggingface",
        device="hpu",
        torch_dtype=torch.bfloat16,
    )
..

**Note:** Intel Neural Compressor will convert the model format from auto-gptq to hpu format on the first load and save hpu_model.safetensors to the local cache directory for the next load. So it may take a while to load for the first time.

Static Quantization with PT2E Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PT2E path uses ``torch.dynamo`` to capture the eager model into an FX graph model, and then inserts the observers and Q/QD pairs on it. Finally it uses the ``torch.compile`` to perform the pattern matching and replace the Q/DQ pairs with optimized quantized operators.

There are four steps to perform W8A8 static quantization with PT2E backend: ``export``, ``prepare``, ``convert`` and ``compile``.

.. code-block:: python

   import torch
   from neural_compressor.torch.export import export
   from neural_compressor.torch.quantization import StaticQuantConfig, prepare, convert

   # Prepare the float model and example inputs for export model
   model = UserFloatModel()
   example_inputs = ...

   # Export eager model into FX graph model
   exported_model = export(model=model, example_inputs=example_inputs)
   # Quantize the model
   quant_config = StaticQuantConfig()
   prepared_model = prepare(exported_model, quant_config=quant_config)
   # Calibrate
   run_fn(prepared_model)
   q_model = convert(prepared_model)
   # Compile the quantized model and replace the Q/DQ pattern with Q-operator
   from torch._inductor import config

   config.freezing = True
   opt_model = torch.compile(q_model)
..

Accuracy-driven Tuning
^^^^^^^^^^^^^^^^^^^^^^

To leverage accuracy-driven automatic tuning, a specified tuning space is necessary. The ``autotune`` iterates the tuning space and applies the configuration on given high-precision model then records and compares its evaluation result with the baseline. The tuning process stops when meeting the exit policy.


.. code-block:: python

   from neural_compressor.torch.quantization import RTNConfig, TuningConfig, autotune


   def eval_fn(model) -> float:
       return ...


   tune_config = TuningConfig(
       config_set=RTNConfig(use_sym=[False, True], group_size=[32, 128]),
       tolerable_loss=0.2,
       max_trials=10,
   )
   q_model = autotune(model, tune_config=tune_config, eval_fn=eval_fn)
..

Tutorials
---------

More detailed tutorials are available in the official Intel® Neural Compressor `doc <https://intel.github.io/neural-compressor/latest/docs/source/Welcome.html>`_.
