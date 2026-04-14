{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "A100",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JTSIV1/emotion-aware-s2st/blob/main/Translation%20LLM.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Part 1: Load API key"
      ],
      "metadata": {
        "id": "UHAhuhdU93zJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q openai transformers sacrebleu pandas\n",
        "!apt-get install -q ffmpeg"
      ],
      "metadata": {
        "id": "Z5SFihKS-XMi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from openai import OpenAI\n",
        "import time, os, random, requests, tarfile, shutil, subprocess\n",
        "from collections import defaultdict, Counter\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "\n",
        "client = OpenAI(\n",
        "    api_key=\"sk-741aba4dc4c64831a5680304db841dd4\",\n",
        "    base_url=\"https://llm.jetstream-cloud.org/api\"\n",
        ")\n",
        "\n",
        "models = client.models.list()\n",
        "for m in models.data:\n",
        "    print(m.id)"
      ],
      "metadata": {
        "id": "fS8qzGuUVnlx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "MODEL_ID = \"meta-llama/Llama-4-Scout-17B-16E-Instruct\"\n",
        "print(f\"Using model: {MODEL_ID}\")"
      ],
      "metadata": {
        "id": "E6E7ZaWJ9sfw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Part 2: Load MELD-ST dataset"
      ],
      "metadata": {
        "id": "U4ZL16DY9__I"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "from IPython.display import display, Javascript\n",
        "\n",
        "display(Javascript('''\n",
        "    function ClickConnect(){\n",
        "        console.log(\"Keeping alive...\");\n",
        "        document.querySelector(\"colab-toolbar-button#connect\").click()\n",
        "    }\n",
        "    setInterval(ClickConnect, 60000)\n",
        "'''))"
      ],
      "metadata": {
        "id": "nR7jbaOoMgFl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "MELD_URL        = \"http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz\"\n",
        "TEMP_TAR        = \"MELD_Raw.tar.gz\"\n",
        "RAW_EXTRACT_DIR = \"./temp_meld_extract\"\n",
        "DATA_DIR        = \"./data\"\n",
        "\n",
        "def setup_directories(base_path):\n",
        "    splits = ['train', 'dev', 'test']\n",
        "    for split in splits:\n",
        "        path = os.path.join(base_path, split)\n",
        "        if not os.path.exists(path):\n",
        "            os.makedirs(path)\n",
        "\n",
        "def download_file(url, dest):\n",
        "    print(\"downloading raw data...\")\n",
        "    response = requests.get(url, stream=True)\n",
        "    with open(dest, 'wb') as f:\n",
        "        for chunk in response.iter_content(chunk_size=1024*1024):\n",
        "            if chunk:\n",
        "                f.write(chunk)\n",
        "    print(\"download complete\")\n",
        "\n",
        "setup_directories(DATA_DIR)\n",
        "download_file(MELD_URL, TEMP_TAR)\n",
        "print(\"Done.\")"
      ],
      "metadata": {
        "id": "rPNmh_LXJAx3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def process_and_cleanup(extract_path, final_data_path):\n",
        "    meld_raw_dir = os.path.join(extract_path, \"MELD.Raw\")\n",
        "    splits = ['train', 'dev', 'test']\n",
        "    exported_name = {\n",
        "        'train': \"train_splits\",\n",
        "        'dev':   \"dev_splits_complete\",\n",
        "        'test':  \"output_repeated_splits_test\"\n",
        "    }\n",
        "\n",
        "    for split in splits:\n",
        "        inner_tar_name = f\"{split}.tar.gz\"\n",
        "        inner_tar_path = os.path.join(meld_raw_dir, inner_tar_name)\n",
        "\n",
        "        if not os.path.exists(inner_tar_path):\n",
        "            print(f\"{inner_tar_name} not found.\")\n",
        "            continue\n",
        "\n",
        "        print(f\"extracting {inner_tar_name}...\")\n",
        "        with tarfile.open(inner_tar_path, \"r:gz\") as tar:\n",
        "            tar.extractall(path=meld_raw_dir)\n",
        "\n",
        "        temp_video_folder = os.path.join(meld_raw_dir, exported_name[split])\n",
        "        output_folder     = os.path.join(final_data_path, split)\n",
        "\n",
        "        print(f\"converting {split} clips to 16kHz audio...\")\n",
        "        if os.path.exists(temp_video_folder):\n",
        "            for filename in os.listdir(temp_video_folder):\n",
        "                if filename.endswith(\".mp4\"):\n",
        "                    video_path = os.path.join(temp_video_folder, filename)\n",
        "                    audio_path = os.path.join(output_folder, filename.replace(\".mp4\", \".wav\"))\n",
        "                    try:\n",
        "                        subprocess.run([\n",
        "                            'ffmpeg', '-y', '-i', video_path,\n",
        "                            '-vn',\n",
        "                            '-acodec', 'pcm_s16le',\n",
        "                            '-ar', '16000',\n",
        "                            '-ac', '1',\n",
        "                            '-loglevel', 'error',\n",
        "                            audio_path\n",
        "                        ], check=True)\n",
        "                        os.remove(video_path)\n",
        "                    except Exception as e:\n",
        "                        print(f\"error processing {filename}: {e}\")\n",
        "            shutil.rmtree(temp_video_folder)\n",
        "\n",
        "        os.remove(inner_tar_path)\n",
        "        print(f\"finished processing and cleaning up {split} set.\")\n",
        "\n",
        "    print(\"Moving CSV label files to ./data...\")\n",
        "    for csv_file in [f for f in os.listdir(meld_raw_dir) if f.endswith('.csv')]:\n",
        "        shutil.move(\n",
        "            os.path.join(meld_raw_dir, csv_file),\n",
        "            os.path.join(final_data_path, csv_file)\n",
        "        )\n",
        "\n",
        "print(\"extracting data...\")\n",
        "with tarfile.open(TEMP_TAR, \"r:gz\") as tar:\n",
        "    tar.extractall(path=RAW_EXTRACT_DIR)\n",
        "os.remove(TEMP_TAR)\n",
        "\n",
        "process_and_cleanup(RAW_EXTRACT_DIR, DATA_DIR)\n",
        "\n",
        "print(\"cleanup...\")\n",
        "shutil.rmtree(RAW_EXTRACT_DIR)\n",
        "print(f\"done! data in {os.path.abspath(DATA_DIR)}\")"
      ],
      "metadata": {
        "id": "ODtgU3ow9ySI",
        "collapsed": true
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Save processed data to Drive immediately after Cell 5 finishes\n",
        "from google.colab import drive\n",
        "\n",
        "# Mount only if not already mounted\n",
        "if not os.path.exists(\"/content/drive\"):\n",
        "    drive.mount('/content/drive')\n",
        "\n",
        "import shutil\n",
        "if not os.path.exists(\"/content/drive/MyDrive/MELD_data\"):\n",
        "    shutil.copytree(\"./data\", \"/content/drive/MyDrive/MELD_data\")\n",
        "    print(\"Saved to Drive!\")\n",
        "else:\n",
        "    print(\"Already saved to Drive, skipping.\")"
      ],
      "metadata": {
        "id": "EQvX5hkfRhuU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_raw = pd.read_csv(os.path.join(DATA_DIR, \"train_sent_emo.csv\"))\n",
        "print(\"Shape:\", df_raw.shape)\n",
        "print(\"\\nColumns:\", df_raw.columns.tolist())\n",
        "df_raw.head(5)"
      ],
      "metadata": {
        "id": "S6Hz3VU_93Bd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Check emotion labels\n",
        "print(\"Raw emotion labels in train set:\")\n",
        "print(df_raw[\"Emotion\"].value_counts())"
      ],
      "metadata": {
        "id": "mnTycrpe-Kpm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Check sample utterances\n",
        "print(\"Sample utterances per emotion:\\n\")\n",
        "for emotion in df_raw[\"Emotion\"].unique():\n",
        "    sample = df_raw[df_raw[\"Emotion\"] == emotion][\"Utterance\"].iloc[0]\n",
        "    print(f\"[{emotion:10s}] {sample}\")"
      ],
      "metadata": {
        "id": "bcBWOMT1-Mea"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Part 3: Emotion mapping"
      ],
      "metadata": {
        "id": "lWayM395-M84"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "EMOTIONS = [\"anger\", \"disgust\", \"fear\", \"joy\", \"neutral\", \"sadness\", \"surprise\"]\n",
        "EMOTION_MAP = {e: e for e in EMOTIONS}\n",
        "\n",
        "print(\"Emotions:\", EMOTIONS)"
      ],
      "metadata": {
        "id": "YwnKzdQO_zAY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def load_meld_split(split):\n",
        "    csv_map = {\n",
        "        \"train\": \"train_sent_emo.csv\",\n",
        "        \"dev\":   \"dev_sent_emo.csv\",\n",
        "        \"test\":  \"test_sent_emo.csv\",\n",
        "    }\n",
        "    de_csv_map = {\n",
        "        \"train\": \"deu_train.csv\",\n",
        "        \"dev\":   \"deu_dev.csv\",\n",
        "        \"test\":  \"deu_test.csv\",\n",
        "    }\n",
        "\n",
        "    df = pd.read_csv(os.path.join(DATA_DIR, csv_map[split]))\n",
        "\n",
        "    # Load DE references\n",
        "    de_path = os.path.join(DATA_DIR, de_csv_map[split])\n",
        "    de_lookup = {}\n",
        "    if os.path.exists(de_path):\n",
        "        df_de = pd.read_csv(de_path)\n",
        "        for _, row in df_de.iterrows():\n",
        "            key = f\"{row['dia_id']}_{row['utt_id']}\"\n",
        "            de_lookup[key] = str(row.get(\"German\", \"\")).strip()\n",
        "        print(f\"Loaded {len(de_lookup)} DE references for {split}\")\n",
        "    else:\n",
        "        print(f\"No DE reference found for {split} — BLEU will be N/A\")\n",
        "\n",
        "    data = []\n",
        "    for _, row in df.iterrows():\n",
        "        emotion = str(row.get(\"Emotion\", \"\")).lower().strip()\n",
        "        if emotion not in EMOTIONS:\n",
        "            continue\n",
        "        en_text = str(row.get(\"Utterance\", \"\")).strip()\n",
        "        if not en_text:\n",
        "            continue\n",
        "        utt_id = f\"{row.get('Dialogue_ID', '')}_{row.get('Utterance_ID', '')}\"\n",
        "        data.append({\n",
        "            \"utterance_id\": utt_id,\n",
        "            \"emotion\":      emotion,\n",
        "            \"en_text\":      en_text,\n",
        "            \"speaker\":      str(row.get(\"Speaker\", \"\")),\n",
        "            \"de_text\":      de_lookup.get(utt_id, None),\n",
        "        })\n",
        "    return data\n",
        "\n",
        "train_data = load_meld_split(\"train\")\n",
        "val_data   = load_meld_split(\"dev\")\n",
        "test_data  = load_meld_split(\"test\")\n",
        "\n",
        "print(f\"\\nTrain: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}\")"
      ],
      "metadata": {
        "id": "wb6qxKx8_8WV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for split_name, split_data in [(\"Train\", train_data), (\"Val\", val_data), (\"Test\", test_data)]:\n",
        "    print(f\"\\n{split_name} emotion distribution:\")\n",
        "    for e, c in sorted(Counter(d[\"emotion\"] for d in split_data).items()):\n",
        "        print(f\"  {e:10s}: {c}\")"
      ],
      "metadata": {
        "id": "otPpF-1X__Dt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Sample entries per emotion from test set:\\n\")\n",
        "for emotion in sorted(set(d[\"emotion\"] for d in test_data)):\n",
        "    sample = next(d for d in test_data if d[\"emotion\"] == emotion)\n",
        "    print(f\"[{emotion:10s}] {sample['en_text']}\")"
      ],
      "metadata": {
        "id": "TXouoyN3AC6V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "print(\"Average word count per emotion (test set):\")\n",
        "for emotion in sorted(set(d[\"emotion\"] for d in test_data)):\n",
        "    subset = [d for d in test_data if d[\"emotion\"] == emotion]\n",
        "    avg_len = np.mean([len(d[\"en_text\"].split()) for d in subset])\n",
        "    print(f\"  {emotion:10s}: {avg_len:.1f} words  (n={len(subset)})\")"
      ],
      "metadata": {
        "id": "ZvlwxFT3BYJ2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "speakers = set(d[\"speaker\"] for d in train_data)\n",
        "print(f\"Unique speakers in train: {len(speakers)}\")\n",
        "print(\"Speakers:\", sorted(speakers))"
      ],
      "metadata": {
        "id": "L0iMevpMBagt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Section 3: Translation EN to DE (German)"
      ],
      "metadata": {
        "id": "y3ey3iWvQ12t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def translate(transcript: str, emotion: str, retries: int = 3) -> dict:\n",
        "    \"\"\"\n",
        "    Input  (from ASR): English transcript + emotion label\n",
        "    Output (to TTS):   {translated_text: German, emotion: str}\n",
        "    \"\"\"\n",
        "    for attempt in range(retries):\n",
        "        try:\n",
        "            response = client.chat.completions.create(\n",
        "                model=MODEL_ID,\n",
        "                messages=[\n",
        "                    {\"role\": \"system\", \"content\": (\n",
        "                        f\"You are an expert English-to-German translator. \"\n",
        "                        f\"The speaker's emotion is: {emotion}. \"\n",
        "                        f\"Use vocabulary, punctuation, and expressions \"\n",
        "                        f\"appropriate for a {emotion} speaker in German. \"\n",
        "                        f\"Output only the German translation, nothing else.\"\n",
        "                    )},\n",
        "                    {\"role\": \"user\", \"content\": f'Translate: \"{transcript}\"'}\n",
        "                ],\n",
        "                temperature=0.3,\n",
        "                max_tokens=256,\n",
        "            )\n",
        "            return {\"translated_text\": response.choices[0].message.content.strip(), \"emotion\": emotion}\n",
        "        except Exception as e:\n",
        "            print(f\"Attempt {attempt+1} failed: {e}\")\n",
        "            if attempt < retries - 1:\n",
        "                time.sleep(2)\n",
        "    return {\"translated_text\": \"\", \"emotion\": emotion}"
      ],
      "metadata": {
        "id": "B_zoPCi_BdQZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def translate_baseline(transcript: str, retries: int = 3) -> dict:\n",
        "    \"\"\"Plain translation with NO emotion conditioning.\"\"\"\n",
        "    for attempt in range(retries):\n",
        "        try:\n",
        "            response = client.chat.completions.create(\n",
        "                model=MODEL_ID,\n",
        "                messages=[\n",
        "                    {\"role\": \"system\", \"content\": (\n",
        "                        \"You are an expert English-to-German translator. \"\n",
        "                        \"Output only the German translation, nothing else.\"\n",
        "                    )},\n",
        "                    {\"role\": \"user\", \"content\": f'Translate: \"{transcript}\"'}\n",
        "                ],\n",
        "                temperature=0.3,\n",
        "                max_tokens=256,\n",
        "            )\n",
        "            return {\"translated_text\": response.choices[0].message.content.strip()}\n",
        "        except Exception as e:\n",
        "            print(f\"Attempt {attempt+1} failed: {e}\")\n",
        "            if attempt < retries - 1:\n",
        "                time.sleep(2)\n",
        "    return {\"translated_text\": \"\"}"
      ],
      "metadata": {
        "id": "EgTjbwXFBiry"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "MODEL_ID = \"llama-4-scout\"\n",
        "print(f\"Using model: {MODEL_ID}\")"
      ],
      "metadata": {
        "id": "X7prOE_9sr-5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Sanity check\n",
        "test_cases = [\n",
        "    (\"I can't believe you did that!\",      \"anger\"),\n",
        "    (\"That is absolutely disgusting.\",     \"disgust\"),\n",
        "    (\"I am so scared right now.\",          \"fear\"),\n",
        "    (\"This is the best day ever!\",         \"joy\"),\n",
        "    (\"Please close the door.\",             \"neutral\"),\n",
        "    (\"I miss you so much.\",                \"sadness\"),\n",
        "    (\"No way, that actually happened?\",    \"surprise\"),\n",
        "]\n",
        "\n",
        "print(\"Sanity Check: EN→DE with emotion conditioning\\n\")\n",
        "for en, emotion in test_cases:\n",
        "    out = translate(en, emotion)\n",
        "    print(f\"[{emotion:10s}] EN: {en}\")\n",
        "    print(f\"             DE: {out['translated_text']}\\n\")"
      ],
      "metadata": {
        "id": "_eAmzyItBnlX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Part 4: Sentiment classifier"
      ],
      "metadata": {
        "id": "IsZhGTczB1au"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import pipeline\n",
        "\n",
        "sentiment_pipe = pipeline(\n",
        "    \"text-classification\",\n",
        "    model=\"lxyuan/distilbert-base-multilingual-cased-sentiments-student\",\n",
        "    device=0,  # use -1 if no GPU\n",
        ")\n",
        "\n",
        "# neutral excluded — 3-class model can't reliably detect it\n",
        "SENTIMENT_TO_EMOTIONS = {\n",
        "    \"positive\": {\"joy\", \"surprise\"},\n",
        "    \"negative\": {\"anger\", \"disgust\", \"fear\", \"sadness\"},\n",
        "}\n",
        "\n",
        "print(\"Sentiment classifier ready!\")"
      ],
      "metadata": {
        "id": "ytwMX43IB1BP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "EMOTION_MARKERS = {\n",
        "    \"anger\":    [\"verdammt\", \"unglaublich\", \"wie kannst du\", \"schäm dich\", \"!\"],\n",
        "    \"disgust\":  [\"ekelhaft\", \"widerlich\", \"abstoßend\", \"igitt\", \"pfui\"],\n",
        "    \"fear\":     [\"angst\", \"fürchte\", \"erschrocken\", \"hilfe\", \"bitte nicht\"],\n",
        "    \"joy\":      [\"wunderbar\", \"toll\", \"super\", \"fantastisch\", \"großartig\", \"hurra\"],\n",
        "    \"neutral\":  [\"ich\", \"das\", \"ist\", \"und\", \"die\", \"der\"],\n",
        "    \"sadness\":  [\"leider\", \"traurig\", \"schade\", \"es tut mir leid\"],\n",
        "    \"surprise\": [\"unglaublich\", \"wirklich\", \"echt\", \"wow\", \"nicht möglich\", \"oh\"],\n",
        "}\n",
        "\n",
        "print(\"German emotion markers defined:\")\n",
        "for e, markers in EMOTION_MARKERS.items():\n",
        "    print(f\"  {e:10s}: {markers}\")"
      ],
      "metadata": {
        "id": "NQh3EZXtB6BO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Quick check that the sentiment classifier works on German text\n",
        "test_sentences = [\n",
        "    (\"Das ist verdammt nochmal unglaublich!\",        \"anger\"),\n",
        "    (\"Das ist ekelhaft, ich kann das nicht ansehen.\", \"disgust\"),\n",
        "    (\"Ich habe so eine Angst, bitte hilf mir.\",       \"fear\"),\n",
        "    (\"Das ist wunderbar, ich bin so glücklich!\",      \"joy\"),\n",
        "    (\"Bitte schließ die Tür.\",                        \"neutral\"),\n",
        "    (\"Ich vermisse dich so sehr, es tut mir leid.\",   \"sadness\"),\n",
        "    (\"Wow, das ist wirklich passiert?\",               \"surprise\"),\n",
        "]\n",
        "\n",
        "print(\"=== Sentiment Classifier Test on German ===\\n\")\n",
        "for text, expected in test_sentences:\n",
        "    pred = sentiment_pipe(text[:512])[0]\n",
        "    print(f\"Text     : {text}\")\n",
        "    print(f\"Expected : {expected}\")\n",
        "    print(f\"Predicted: {pred['label']} (score: {pred['score']:.2f})\\n\")"
      ],
      "metadata": {
        "id": "bFacklGvB738"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 20 per emotion = 140 items total (7 emotions × 20)\n",
        "small_test = []\n",
        "for emotion in sorted(set(d[\"emotion\"] for d in test_data)):\n",
        "    subset = [d for d in test_data if d[\"emotion\"] == emotion][:20]\n",
        "    small_test += subset\n",
        "\n",
        "print(f\"Small test set: {len(small_test)} items\")\n",
        "for e, c in sorted(Counter(d[\"emotion\"] for d in small_test).items()):\n",
        "    print(f\"  {e:10s}: {c}\")"
      ],
      "metadata": {
        "id": "1xtJhMubB9nK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Part 5: Evaluation"
      ],
      "metadata": {
        "id": "mFhZTw0dCDdd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_emotion_conditioned(data):\n",
        "    rows = []\n",
        "    for emotion in sorted(set(d[\"emotion\"] for d in data)):\n",
        "        subset = [d for d in data if d[\"emotion\"] == emotion]\n",
        "        sent_correct, marker_count, sent_total = 0, 0, 0\n",
        "\n",
        "        for item in subset:\n",
        "            result = translate(item[\"en_text\"], item[\"emotion\"])\n",
        "            de = result[\"translated_text\"]\n",
        "            if not de:\n",
        "                continue\n",
        "\n",
        "            if emotion != \"neutral\":\n",
        "                pred     = sentiment_pipe(de[:512])[0][\"label\"].lower()\n",
        "                expected = {k for k, v in SENTIMENT_TO_EMOTIONS.items() if emotion in v}\n",
        "                sent_correct += int(pred in expected)\n",
        "                sent_total   += 1\n",
        "\n",
        "            marker_count += int(any(m in de.lower() for m in EMOTION_MARKERS[emotion]))\n",
        "\n",
        "        rows.append({\n",
        "            \"Emotion\":        emotion,\n",
        "            \"N\":              len(subset),\n",
        "            \"Sentiment acc.\": f\"{sent_correct/sent_total:.1%}\" if sent_total else \"N/A\",\n",
        "            \"Marker rate\":    f\"{marker_count/len(subset):.1%}\",\n",
        "        })\n",
        "\n",
        "    df = pd.DataFrame(rows)\n",
        "    print(\"\\n=== Emotion-Conditioned Translation Results ===\")\n",
        "    print(df.to_string(index=False))\n",
        "    return df\n",
        "\n",
        "conditioned_df = evaluate_emotion_conditioned(small_test)"
      ],
      "metadata": {
        "id": "xmZQynTuB_eR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_emotion_conditioned(data):\n",
        "    results = []\n",
        "    for emotion in sorted(set(d[\"emotion\"] for d in data)):\n",
        "        subset = [d for d in data if d[\"emotion\"] == emotion]\n",
        "        sent_correct, marker_count, sent_total = 0, 0, 0\n",
        "\n",
        "        for item in subset:\n",
        "            result = translate(item[\"en_text\"], item[\"emotion\"])\n",
        "            de = result[\"translated_text\"]\n",
        "            if not de:\n",
        "                continue\n",
        "\n",
        "            if emotion != \"neutral\":\n",
        "                pred     = sentiment_pipe(de[:512])[0][\"label\"].lower()\n",
        "                expected = {k for k, v in SENTIMENT_TO_EMOTIONS.items() if emotion in v}\n",
        "                sent_correct += int(pred in expected)\n",
        "                sent_total   += 1\n",
        "\n",
        "            marker_count += int(any(m in de.lower() for m in EMOTION_MARKERS[emotion]))\n",
        "\n",
        "        results.append({\n",
        "            \"emotion\":    emotion,\n",
        "            \"n\":          len(subset),\n",
        "            \"sent_acc\":   sent_correct / sent_total if sent_total else None,\n",
        "            \"marker_rate\": marker_count / len(subset),\n",
        "        })\n",
        "    return results\n",
        "\n",
        "conditioned_results = evaluate_emotion_conditioned(small_test)\n",
        "print(\"Done! Results collected.\")"
      ],
      "metadata": {
        "id": "_5RaG224wqOi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "rows = []\n",
        "for r in conditioned_results:\n",
        "    rows.append({\n",
        "        \"Emotion\":        r[\"emotion\"],\n",
        "        \"N\":              r[\"n\"],\n",
        "        \"Sentiment acc.\": f\"{r['sent_acc']:.1%}\" if r[\"sent_acc\"] is not None else \"N/A\",\n",
        "        \"Marker rate\":    f\"{r['marker_rate']:.1%}\",\n",
        "    })\n",
        "\n",
        "conditioned_df = pd.DataFrame(rows)\n",
        "print(\"\\n=== Emotion-Conditioned Translation Results ===\")\n",
        "print(conditioned_df.to_string(index=False))"
      ],
      "metadata": {
        "id": "9QZ2g1NRwvSk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "emotions    = [r[\"emotion\"] for r in conditioned_results]\n",
        "sent_vals   = [r[\"sent_acc\"] * 100 if r[\"sent_acc\"] is not None else 0 for r in conditioned_results]\n",
        "marker_vals = [r[\"marker_rate\"] * 100 for r in conditioned_results]\n",
        "\n",
        "x     = np.arange(len(emotions))\n",
        "width = 0.35\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "b1 = ax.bar(x - width/2, sent_vals,   width, label='Sentiment Acc.', color='steelblue', edgecolor='white')\n",
        "b2 = ax.bar(x + width/2, marker_vals, width, label='Marker Rate',    color='coral',     edgecolor='white')\n",
        "ax.bar_label(b1, fmt='%.1f%%', padding=3, fontsize=8)\n",
        "ax.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=8)\n",
        "ax.set_title(\"Emotion-Conditioned Translation Results\")\n",
        "ax.set_ylabel(\"Score (%)\")\n",
        "ax.set_ylim(0, 115)\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(emotions)\n",
        "ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig(\"./conditioned_results.png\", dpi=150)\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "KQfuFUJTwxwY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def full_comparison(data):\n",
        "    rows = []\n",
        "    for emotion in sorted(set(d[\"emotion\"] for d in data)):\n",
        "        subset = [d for d in data if d[\"emotion\"] == emotion]\n",
        "        base_sent, prompt_sent = 0, 0\n",
        "        base_mark, prompt_mark = 0, 0\n",
        "        sent_total, total = 0, 0\n",
        "\n",
        "        for item in subset:\n",
        "            base_de   = translate_baseline(item[\"en_text\"])[\"translated_text\"]\n",
        "            prompt_de = translate(item[\"en_text\"], item[\"emotion\"])[\"translated_text\"]\n",
        "            if not base_de or not prompt_de:\n",
        "                continue\n",
        "            total += 1\n",
        "\n",
        "            if emotion != \"neutral\":\n",
        "                expected    = {k for k, v in SENTIMENT_TO_EMOTIONS.items() if emotion in v}\n",
        "                base_pred   = sentiment_pipe(base_de[:512])[0][\"label\"].lower()\n",
        "                prompt_pred = sentiment_pipe(prompt_de[:512])[0][\"label\"].lower()\n",
        "                base_sent   += int(base_pred in expected)\n",
        "                prompt_sent += int(prompt_pred in expected)\n",
        "                sent_total  += 1\n",
        "\n",
        "            base_mark   += int(any(m in base_de.lower()   for m in EMOTION_MARKERS[emotion]))\n",
        "            prompt_mark += int(any(m in prompt_de.lower() for m in EMOTION_MARKERS[emotion]))\n",
        "\n",
        "        rows.append({\n",
        "            \"Emotion\":         emotion,\n",
        "            \"N\":               total,\n",
        "            \"Base sent.\":      f\"{base_sent/sent_total:.1%}\"   if sent_total else \"N/A\",\n",
        "            \"Prompted sent.\":  f\"{prompt_sent/sent_total:.1%}\" if sent_total else \"N/A\",\n",
        "            \"Base marker\":     f\"{base_mark/total:.1%}\",\n",
        "            \"Prompted marker\": f\"{prompt_mark/total:.1%}\",\n",
        "        })\n",
        "\n",
        "    df = pd.DataFrame(rows)\n",
        "    print(\"\\n=== Baseline vs Emotion-Prompted (EN→DE) ===\")\n",
        "    print(df.to_string(index=False))\n",
        "    print(\"\\nHigher prompted scores = emotion conditioning is working.\")\n",
        "    return df\n",
        "\n",
        "comparison_df = full_comparison(small_test)"
      ],
      "metadata": {
        "id": "VkJOoNTICBZL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate DE pseudo-references for small_test and save\n",
        "import json\n",
        "\n",
        "CACHE_PATH = \"./data/de_pseudo_references.json\"\n",
        "\n",
        "# Load cache if exists\n",
        "if os.path.exists(CACHE_PATH):\n",
        "    with open(CACHE_PATH, \"r\", encoding=\"utf-8\") as f:\n",
        "        de_cache = json.load(f)\n",
        "    print(f\"Loaded {len(de_cache)} cached DE references.\")\n",
        "else:\n",
        "    de_cache = {}\n",
        "\n",
        "# Generate for items not yet cached\n",
        "for item in small_test:\n",
        "    key = f\"{item['utterance_id']}_{item['emotion']}\"\n",
        "    if key not in de_cache:\n",
        "        result = translate(item[\"en_text\"], item[\"emotion\"])\n",
        "        de_cache[key] = result[\"translated_text\"]\n",
        "\n",
        "# Save cache\n",
        "with open(CACHE_PATH, \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(de_cache, f, ensure_ascii=False)\n",
        "print(f\"Saved {len(de_cache)} DE references.\")\n",
        "\n",
        "# Attach to small_test\n",
        "for item in small_test:\n",
        "    key = f\"{item['utterance_id']}_{item['emotion']}\"\n",
        "    item[\"de_text\"] = de_cache.get(key, None)\n",
        "\n",
        "print(\"DE references attached to small_test.\")"
      ],
      "metadata": {
        "id": "kHdc5QXPudR3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import sacrebleu\n",
        "\n",
        "def evaluate_bleu(data):\n",
        "    subset_with_de = [d for d in data if d.get(\"de_text\")]\n",
        "    if not subset_with_de:\n",
        "        print(\"No DE references available — skipping BLEU.\")\n",
        "        return\n",
        "\n",
        "    by_emotion = defaultdict(lambda: {\"hyps\": [], \"refs\": []})\n",
        "    for item in subset_with_de:\n",
        "        result = translate(item[\"en_text\"], item[\"emotion\"])\n",
        "        by_emotion[item[\"emotion\"]][\"hyps\"].append(result[\"translated_text\"])\n",
        "        by_emotion[item[\"emotion\"]][\"refs\"].append(item[\"de_text\"])\n",
        "\n",
        "    print(\"\\n=== BLEU Scores per Emotion ===\")\n",
        "    all_hyps, all_refs = [], []\n",
        "    for emotion in sorted(by_emotion):\n",
        "        bleu = sacrebleu.corpus_bleu(\n",
        "            by_emotion[emotion][\"hyps\"],\n",
        "            [by_emotion[emotion][\"refs\"]]\n",
        "        )\n",
        "        print(f\"  {emotion:10s}: BLEU = {bleu.score:.2f}\")\n",
        "        all_hyps += by_emotion[emotion][\"hyps\"]\n",
        "        all_refs += by_emotion[emotion][\"refs\"]\n",
        "\n",
        "    overall = sacrebleu.corpus_bleu(all_hyps, [all_refs])\n",
        "    print(f\"  {'Overall':10s}: BLEU = {overall.score:.2f}\")\n",
        "\n",
        "evaluate_bleu(small_test)"
      ],
      "metadata": {
        "id": "FYoKPE-xCI2V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_chrf(data):\n",
        "    by_emotion = defaultdict(lambda: {\"hyps\": [], \"refs\": []})\n",
        "    for item in data:\n",
        "        result = translate(item[\"en_text\"], item[\"emotion\"])\n",
        "        by_emotion[item[\"emotion\"]][\"hyps\"].append(result[\"translated_text\"])\n",
        "        by_emotion[item[\"emotion\"]][\"refs\"].append(item[\"de_text\"])\n",
        "\n",
        "    print(\"\\n=== chrF Scores per Emotion ===\")\n",
        "    all_hyps, all_refs = [], []\n",
        "    for emotion in sorted(by_emotion):\n",
        "        chrf = sacrebleu.corpus_chrf(\n",
        "            by_emotion[emotion][\"hyps\"],\n",
        "            [by_emotion[emotion][\"refs\"]]\n",
        "        )\n",
        "        print(f\"  {emotion:10s}: chrF = {chrf.score:.2f}\")\n",
        "        all_hyps += by_emotion[emotion][\"hyps\"]\n",
        "        all_refs += by_emotion[emotion][\"refs\"]\n",
        "\n",
        "    overall = sacrebleu.corpus_chrf(all_hyps, [all_refs])\n",
        "    print(f\"  {'Overall':10s}: chrF = {overall.score:.2f}\")\n",
        "\n",
        "evaluate_chrf(small_test)"
      ],
      "metadata": {
        "id": "2JK-9xX0EAK2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q bert-score\n",
        "\n",
        "from bert_score import score as bert_score\n",
        "\n",
        "def evaluate_bertscore(data):\n",
        "    \"\"\"\n",
        "    Compares emotion-prompted vs baseline translations semantically.\n",
        "    Higher BERTScore = prompted output is more semantically rich/different.\n",
        "    Uses multilingual BERT to handle German.\n",
        "    \"\"\"\n",
        "    prompted_outputs, baseline_outputs = [], []\n",
        "\n",
        "    for item in data:\n",
        "        prompted = translate(item[\"en_text\"], item[\"emotion\"])[\"translated_text\"]\n",
        "        baseline = translate_baseline(item[\"en_text\"])[\"translated_text\"]\n",
        "        if prompted and baseline:\n",
        "            prompted_outputs.append(prompted)\n",
        "            baseline_outputs.append(baseline)\n",
        "\n",
        "    P, R, F1 = bert_score(\n",
        "        prompted_outputs,\n",
        "        baseline_outputs,\n",
        "        lang=\"de\",\n",
        "        verbose=False\n",
        "    )\n",
        "\n",
        "    print(\"\\n=== BERTScore: Emotion-Prompted vs Baseline ===\")\n",
        "    print(f\"  Precision : {P.mean():.4f}\")\n",
        "    print(f\"  Recall    : {R.mean():.4f}\")\n",
        "    print(f\"  F1        : {F1.mean():.4f}\")\n",
        "    print(\"\\nNote: Lower F1 = more difference between prompted and baseline\")\n",
        "    print(\"      which suggests emotion conditioning is changing the output.\")\n",
        "\n",
        "evaluate_bertscore(small_test)"
      ],
      "metadata": {
        "id": "f5EGQsgxECCz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_emotion_consistency(data, n_samples=5):\n",
        "    \"\"\"\n",
        "    Translates same sentence across all 5 emotions.\n",
        "    Checks if outputs are actually different — proves emotion conditioning works.\n",
        "    \"\"\"\n",
        "    sample_sentences = [d[\"en_text\"] for d in data[:n_samples]]\n",
        "    emotions = [\"angry\", \"happy\", \"neutral\", \"sad\", \"surprise\"]\n",
        "\n",
        "    print(\"=== Emotion Consistency Check ===\")\n",
        "    print(\"Same sentence translated with different emotion labels:\\n\")\n",
        "\n",
        "    for sentence in sample_sentences:\n",
        "        print(f\"EN: {sentence}\")\n",
        "        translations = {}\n",
        "        for emotion in emotions:\n",
        "            de = translate(sentence, emotion)[\"translated_text\"]\n",
        "            translations[emotion] = de\n",
        "            print(f\"  [{emotion:8s}]: {de}\")\n",
        "\n",
        "        # Check uniqueness — are all 5 outputs different?\n",
        "        unique = len(set(translations.values()))\n",
        "        print(f\"  → Unique translations: {unique}/5\\n\")\n",
        "\n",
        "evaluate_emotion_consistency(small_test)"
      ],
      "metadata": {
        "id": "8yRzgg2PED5M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def full_summary(data):\n",
        "    rows = []\n",
        "    for emotion in sorted(set(d[\"emotion\"] for d in data)):\n",
        "        subset = [d for d in data if d[\"emotion\"] == emotion]\n",
        "        sent_correct, marker_count = 0, 0\n",
        "        sent_total = 0\n",
        "        bleu_hyps, bleu_refs = [], []\n",
        "        chrf_hyps, chrf_refs = [], []\n",
        "\n",
        "        for item in subset:\n",
        "            result = translate(item[\"en_text\"], item[\"emotion\"])\n",
        "            de = result[\"translated_text\"]\n",
        "            if not de:\n",
        "                continue\n",
        "\n",
        "            # Sentiment\n",
        "            if emotion != \"neutral\":\n",
        "                pred     = sentiment_pipe(de[:512])[0][\"label\"].lower()\n",
        "                expected = {k for k, v in SENTIMENT_TO_EMOTIONS.items() if emotion in v}\n",
        "                sent_correct += int(pred in expected)\n",
        "                sent_total   += 1\n",
        "\n",
        "            # Marker\n",
        "            marker_count += int(any(m in de.lower() for m in EMOTION_MARKERS[emotion]))\n",
        "\n",
        "            # BLEU + chrF (if DE reference available)\n",
        "            if \"de_text\" in item:\n",
        "                bleu_hyps.append(de)\n",
        "                bleu_refs.append(item[\"de_text\"])\n",
        "                chrf_hyps.append(de)\n",
        "                chrf_refs.append(item[\"de_text\"])\n",
        "\n",
        "        bleu_score = sacrebleu.corpus_bleu(bleu_hyps, [bleu_refs]).score if bleu_hyps else None\n",
        "        chrf_score = sacrebleu.corpus_chrf(chrf_hyps, [chrf_refs]).score if chrf_hyps else None\n",
        "\n",
        "        rows.append({\n",
        "            \"Emotion\":        emotion,\n",
        "            \"N\":              len(subset),\n",
        "            \"Sentiment acc.\": f\"{sent_correct/sent_total:.1%}\" if sent_total else \"N/A\",\n",
        "            \"Marker rate\":    f\"{marker_count/len(subset):.1%}\",\n",
        "            \"BLEU\":           f\"{bleu_score:.1f}\" if bleu_score is not None else \"N/A\",\n",
        "            \"chrF\":           f\"{chrf_score:.1f}\" if chrf_score is not None else \"N/A\",\n",
        "        })\n",
        "\n",
        "    df = pd.DataFrame(rows)\n",
        "    print(\"\\n=== Full Evaluation Summary ===\")\n",
        "    print(df.to_string(index=False))\n",
        "    return df\n",
        "\n",
        "summary_df = full_summary(small_test)"
      ],
      "metadata": {
        "id": "GHaxqQ1KEF0M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import shutil, os\n",
        "from google.colab import drive\n",
        "\n",
        "if not os.path.exists(\"/content/drive\"):\n",
        "    drive.mount('/content/drive')\n",
        "\n",
        "# Save processed data\n",
        "if not os.path.exists(\"/content/drive/MyDrive/MELD_data\"):\n",
        "    shutil.copytree(\"./data\", \"/content/drive/MyDrive/MELD_data\")\n",
        "    print(\"Data saved to Drive!\")\n",
        "else:\n",
        "    print(\"Data already in Drive.\")\n",
        "\n",
        "# Save result graphs\n",
        "for f in [\"conditioned_results.png\", \"comparison_results.png\", \"bleu_results.png\", \"chrf_results.png\", \"evaluation_results.png\"]:\n",
        "    if os.path.exists(f\"./{f}\"):\n",
        "        shutil.copy(f\"./{f}\", f\"/content/drive/MyDrive/{f}\")\n",
        "        print(f\"Saved {f} to Drive.\")"
      ],
      "metadata": {
        "id": "1r3EaxEszGHd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "\n",
        "notebook_path = \"/content/drive/MyDrive/LLM_Speech_Tech.ipynb\"  # adjust path\n",
        "\n",
        "with open(notebook_path, \"r\") as f:\n",
        "    nb = json.load(f)\n",
        "\n",
        "# Remove problematic widget metadata\n",
        "if \"widgets\" in nb.get(\"metadata\", {}):\n",
        "    del nb[\"metadata\"][\"widgets\"]\n",
        "\n",
        "with open(notebook_path, \"w\") as f:\n",
        "    json.dump(nb, f)\n",
        "\n",
        "print(\"Fixed! Re-download and push to GitHub.\")"
      ],
      "metadata": {
        "id": "bJh-b2z8zVpi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "\n",
        "# Load your notebook — adjust filename\n",
        "notebook_path = \"/content/drive/MyDrive/LLM_Speech_Tech.ipynb\"\n",
        "\n",
        "with open(notebook_path, \"r\") as f:\n",
        "    nb = json.load(f)\n",
        "\n",
        "# Remove widget metadata from top level\n",
        "if \"widgets\" in nb.get(\"metadata\", {}):\n",
        "    del nb[\"metadata\"][\"widgets\"]\n",
        "\n",
        "# Remove widget metadata from each cell too\n",
        "for cell in nb.get(\"cells\", []):\n",
        "    if \"metadata\" in cell:\n",
        "        if \"widgets\" in cell[\"metadata\"]:\n",
        "            del cell[\"metadata\"][\"widgets\"]\n",
        "    # Clean output metadata\n",
        "    for output in cell.get(\"outputs\", []):\n",
        "        if \"metadata\" in output:\n",
        "            if \"widgets\" in output[\"metadata\"]:\n",
        "                del output[\"metadata\"][\"widgets\"]\n",
        "\n",
        "with open(notebook_path, \"w\") as f:\n",
        "    json.dump(nb, f, indent=1)\n",
        "\n",
        "print(\"Cleaned! Re-download and push.\")"
      ],
      "metadata": {
        "id": "wlZub8xtzulD"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}