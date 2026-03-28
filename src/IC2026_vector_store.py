import os
import torch
import faiss
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.f_utils import load_config, load_models, load_dataset
from PIL import Image


def load_embeddings(config, type_='image'):
    """
    Carrega arquivo .pt
    """
    
    emb_file = os.path.join(config['folders']['embeddings_dir'], config['files'][type_])
    
    if not os.path.exists(emb_file):
        print(f"❌ Arquivo não encontrado: {emb_file}")
        return None
    
    try:
        # Carregar o arquivo torch
        embeddings = torch.load(emb_file)
        
        print(f"✅ Embeddings carregados com sucesso!")
        print(f"📊 Formato dos dados: {type(embeddings)}")
        print(f"📊 Shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Erro ao carregar embeddings: {e}")
        return None
    
def create_vector_store(
        embeddings,
        index_file,
        vector_store_dir
    ) -> faiss.Index:
    """
    Cria um índice Faiss a partir dos embeddings o fornecidos
    """
    
    # Criar vector store FAISS com os embeddings carregados
    print("🔧 Criando vector store FAISS...")

    # Verificar se os embeddings foram carregados corretamente
    if embeddings is not None:
        print(f"📊 Número de embeddings: {embeddings.shape[0]}")
        print(f"📊 Dimensionalidade: {embeddings.shape[1]}")
    
        # Criar FAISS index diretamente
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product para similaridade de cosseno

        # Normalizar embeddings para usar inner product como similaridade de cosseno
        faiss.normalize_L2(embeddings)

        # Adicionar embeddings ao index
        index.add(embeddings)

        # Salva vector store 
        INDEX_FILE = os.path.join(vector_store_dir, index_file)
        faiss.write_index(index, INDEX_FILE)

        print("✅ Vector store criado com sucesso!")
        print(f"📄 Arquivo de índice: {INDEX_FILE}")
        print(f"📊 Total de vetores no índice: {index.ntotal}")
    
    else:
        print("❌ Erro: Embeddings não carregados corretamente")


def main(config_path):

    config = load_config(config_path)

    img_embs = load_embeddings(config, type_='image')
    img_embs = img_embs.numpy()
    txt_embs = load_embeddings(config, type_='text')
    txt_embs = txt_embs.numpy()

    create_vector_store(img_embs, "IC2026_image_faiss.index", config['folders']['vector_store_dir'])
    create_vector_store(txt_embs, "IC2026_text_faiss.index", config['folders']['vector_store_dir'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cria vector store ImageCLEF 2026"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/IC2026_vector_store.yaml",
        help="Caminho para arquivo de configuração"
    )

    args = parser.parse_args()
    main(args.config)