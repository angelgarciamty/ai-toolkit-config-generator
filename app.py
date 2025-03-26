from flask import Flask, render_template, request, jsonify
import yaml
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_yaml():
    # Capturar datos del formulario
    data = request.form.to_dict()
    
    # Crear directorio de workspace si no existe
    workspace_dir = '/workspace/ai-toolkit/config'
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Generar nombre de archivo basado en el nombre del proyecto
    filename = f"{data.get('name', 'default')}_config.yaml"
    filepath = os.path.join(workspace_dir, filename)

    resolutions = data.get('resolution', '768')  # Valor por defecto si no se envía
    resolution_list = [int(r.strip()) for r in resolutions.split(',')]

    prompts = data.get('prompts', '').replace('name', data.get('name', 'default'))
    finalprompts = prompts.splitlines()

    
    # Estructura base de configuración
    config = {
        'job': 'extension',
        'config': {
            'name': data.get('name', 'default'),
            'process': [{
                'type': 'sd_trainer',
                'training_folder': 'output',
                'device': data.get('device', 'cuda:0'),
                'network': {
                    'type': 'lora',
                    'linear': int(data.get('linear', 16)),
                    'linear_alpha': int(data.get('linear_alpha', 16))
                },
                'save': {
                    'dtype': data.get('save_dtype', 'float16'),
                    'save_every': int(data.get('save_every', 240)),
                    'max_step_saves_to_keep': int(data.get('max_step_saves', 4)),
                    'push_to_hub': data.get('push_to_hub', 'false') == 'true'
                },
                'datasets': [{
                    'folder_path': data.get('dataset_path', '/workspace/ai-toolkit/dataset'),
                    'caption_ext': data.get('caption_ext', 'txt'),
                    'caption_dropout_rate': float(data.get('caption_dropout', 0.05)),
                    'shuffle_tokens': data.get('shuffle_tokens', 'false') == 'true',
                    'cache_latents_to_disk': data.get('cache_latents', 'true') == 'true',
                    'resolution': resolution_list
                }],
                'train': {
                    'batch_size': int(data.get('batch_size', 1)),
                    'steps': int(data.get('train_steps', 2400)),
                    'gradient_accumulation_steps': int(data.get('gradient_steps', 1)),
                    'train_unet': data.get('train_unet', 'true') == 'true',
                    'train_text_encoder': data.get('train_text_encoder', 'false') == 'true',
                    'gradient_checkpointing': data.get('gradient_checkpointing', 'true') == 'true',
                    'noise_scheduler': data.get('noise_scheduler', 'flowmatch'),
                    'optimizer': data.get('optimizer', 'adamw8bit'),
                    'lr': float(data.get('learning_rate', 0.0001)),
                    'ema_config': {
                        'use_ema': data.get('use_ema', 'true') == 'true',
                        'ema_decay': float(data.get('ema_decay', 0.99))
                    },
                    'dtype': data.get('train_dtype', 'bf16')
                },
                'model': {
                    'name_or_path': data.get('model_path', 'black-forest-labs/FLUX.1-dev'),
                    'is_flux': data.get('is_flux', 'true') == 'true',
                    'quantize': data.get('quantize', 'true') == 'true'
                },
                'sample': {
                    'sampler': data.get('sampler', 'flowmatch'),
                    'sample_every': int(data.get('sample_every', 250)),
                    'width': int(data.get('width', 768)),
                    'height': int(data.get('height', 768)),
                    'prompts':finalprompts,
                    'neg': data.get('negative_prompt', ''),
                    'seed': int(data.get('seed', 42)),
                    'walk_seed': data.get('walk_seed', 'true') == 'true',
                    'guidance_scale': float(data.get('guidance_scale', 4)),
                    'sample_steps': int(data.get('sample_steps', 20))
                }
            }]
        },
        'meta': {
            'name': data.get('name', 'default'),
            'version': data.get('version', '1.0')
        }
    }
    
    with open(filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    # Devolver mensaje de éxito
    return jsonify({
        'message': f'Archivo de configuración guardado en {filepath}',
        'path': filepath
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)