'''
Este notebook define um logger com uma configuração padrão de mensagens de log.
As mensagens de log são utilizadas para acompanhar o comportamento da aplicação, erros e eventos. Elas facilitam o processo de depuração, fornecem visibilidade sobre o fluxo do programa e auxiliam no monitoramento e diagnóstico de problemas.
O uso de logging melhora a qualidade do código, a manutenção e a resolução de erros.
'''

# Importando bibliotecas
import os
from datetime import datetime
import logging

## O nome do arquivo para o log, incluindo hora e data atual
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

## O caminho para o diretório onde os arquivos de log estão salvos
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

## Cria os diretórios se não existirem
os.makedirs(logs_path,exist_ok=True)


## Caminho completo para o arquivo de log
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)


## Setup de configurações de logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)