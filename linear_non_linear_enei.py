{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "linear_non_linear_enei.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyNOhdxr2T+jC6Zk08ImYAsR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/datascienceayacuchoperu/ML_PROJECTS/blob/main/linear_non_linear_enei.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Ejemplo de modelos lineales y no lineales. Fuente [video de ENEI]([C:\\Users\\PREDATOR\\Documents\\Personal_Website\\academic_webpage\\proyectos_posts\\post\\project_22)"
      ],
      "metadata": {
        "id": "gSYu5mHeDJPG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "5ROn_TexCsyS"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()"
      ],
      "metadata": {
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgZG8gewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwoKICAgICAgbGV0IHBlcmNlbnREb25lID0gZmlsZURhdGEuYnl0ZUxlbmd0aCA9PT0gMCA/CiAgICAgICAgICAxMDAgOgogICAgICAgICAgTWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCk7CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPSBgJHtwZXJjZW50RG9uZX0lIGRvbmVgOwoKICAgIH0gd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCk7CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "nWraHOJODrA_",
        "outputId": "ddc7f0aa-8b90-45a0-e720-f79583413c95"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-3e2d2de5-fbf3-4acf-aa63-ce0f129e65c0\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-3e2d2de5-fbf3-4acf-aa63-ce0f129e65c0\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving Position_Salaries.csv to Position_Salaries.csv\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ""
      ],
      "metadata": {
        "id": "PIPkQ7s9DHAH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "datos = pd.read_csv(\"Position_Salaries.csv\", error_bad_lines = False)\n",
        "datos"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 392
        },
        "id": "dqoSxbsKDGwX",
        "outputId": "50d5b2e3-1a80-4654-80e1-1979837735e8"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: FutureWarning: The error_bad_lines argument has been deprecated and will be removed in a future version.\n",
            "\n",
            "\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "            Position  Level   Salary\n",
              "0   Business Analyst      1    45000\n",
              "1  Junior Consultant      2    50000\n",
              "2  Senior Consultant      3    60000\n",
              "3            Manager      4    80000\n",
              "4    Country Manager      5   110000\n",
              "5     Region Manager      6   150000\n",
              "6            Partner      7   200000\n",
              "7     Senior Partner      8   300000\n",
              "8            C-level      9   500000\n",
              "9                CEO     10  1000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-de5113fb-1b32-4cbf-874b-a2c1b55bf9a2\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Position</th>\n",
              "      <th>Level</th>\n",
              "      <th>Salary</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Business Analyst</td>\n",
              "      <td>1</td>\n",
              "      <td>45000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Junior Consultant</td>\n",
              "      <td>2</td>\n",
              "      <td>50000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Senior Consultant</td>\n",
              "      <td>3</td>\n",
              "      <td>60000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Manager</td>\n",
              "      <td>4</td>\n",
              "      <td>80000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Country Manager</td>\n",
              "      <td>5</td>\n",
              "      <td>110000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Region Manager</td>\n",
              "      <td>6</td>\n",
              "      <td>150000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Partner</td>\n",
              "      <td>7</td>\n",
              "      <td>200000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Senior Partner</td>\n",
              "      <td>8</td>\n",
              "      <td>300000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>C-level</td>\n",
              "      <td>9</td>\n",
              "      <td>500000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>CEO</td>\n",
              "      <td>10</td>\n",
              "      <td>1000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-de5113fb-1b32-4cbf-874b-a2c1b55bf9a2')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-de5113fb-1b32-4cbf-874b-a2c1b55bf9a2 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-de5113fb-1b32-4cbf-874b-a2c1b55bf9a2');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from matplotlib import pyplot as plt\n",
        "X = datos['Level'].values.reshape(-1,1)\n",
        "#se necesita un 2D array para SKlearn\n",
        "y = datos['Salary'].values.reshape(-1,1)\n",
        "plt.scatter(X,y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 293
        },
        "id": "xt0mNA9xEJ5e",
        "outputId": "91085aa3-abfd-421a-8604-6f2cc663e6f4"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7ff19c9c2910>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "model = LinearRegression()\n",
        "model.fit(X,y)\n",
        "y_pred = model.predict(X)\n",
        "y_pred"
      ],
      "metadata": {
        "id": "uxISE9oaGDps"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(X,y)\n",
        "plt.plot(X,y_pred, color = 'b')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "p2EPnzYIGb5H"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Evaluamos el modelo\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "rmse = np.sqrt(mean_squared_error(y,y_pred))\n",
        "r2 = r2_score(y,y_pred)\n",
        "print('RMSE: ' + str(rmse))\n",
        "print('R2: ' + str(r2))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4WjoqgvZGtCR",
        "outputId": "76f3108d-0450-4e8b-9997-37fdf1865603"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "RMSE: 163388.73519272613\n",
            "R2: 0.6690412331929895\n"
          ]
        }
      ]
    }
  ]
}