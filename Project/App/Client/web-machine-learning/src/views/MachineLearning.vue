<template>
  <div>
    <v-container>
      <v-row align="center" justify="center">
        <v-col cols="12" align-self="center" v-if="file == null">
          <div class="title mb-1">Sélectionné un vêtement</div>
        </v-col>
        <v-col cols="12" align-self="center" v-else>
          <div class="title mb-1">IA Vêtement</div>
          <v-img v-if="file != null" :src="file" aspect-ratio="2" contain max-height="350"></v-img>
        </v-col>
        <v-col cols="12">
          <v-file-input @change="downloadFile" accept="image/*" label="Image de vétement" prepend-icon="mdi-camera"></v-file-input>
        </v-col>
        <v-col cols="6">
          <v-file-input @change="downloadJsonFile" label="Modéle pré-entrainé MLP" accept=".json" outlined dense></v-file-input>
        </v-col>
        <v-col cols="6">
          <v-text-field
            label="Lib Machine Learning (MLP)"
            :value="mlpValue"
            v-model="mlpValue"
            outlined
            readonly
          ></v-text-field>
        </v-col>
        <v-col cols="6">
          <v-btn @click="sendJsonFile" block rounded color="primary">Charger le modéle MLP</v-btn>
        </v-col>
        <v-col cols="6">
          <v-btn @click="sendImageMLP" block rounded color="primary">Lancer la prédiction MLP</v-btn>
        </v-col>
        <v-col cols="6">
          <v-file-input @change="downloadJsonFileModelLineaire" label="Modéle pré-entrainé modele lineaire" accept=".json" outlined dense></v-file-input>
        </v-col>
        <v-col cols="6">
          <v-text-field
            label="Lib Machine Learning (modele lineaire)"
            :value="lineareModelValue"
            v-model="lineareModelValue"
            outlined
            readonly
          ></v-text-field>
        </v-col>
        <v-col cols="6">
          <v-combobox
            v-model="selectFirstClass"
            :items="items"
            label="Séléctionner la premiere classe"
          ></v-combobox>
        </v-col>
        <v-col cols="6">
          <v-combobox
            v-model="selectSecondClass"
            :items="items"
            label="Séléctionner la seconde classe"
          ></v-combobox>
        </v-col>
        <v-col cols="6">
          <v-btn @click="sendJsonFileModeleLineaire" block rounded color="warning">Charger le modéle lineaire</v-btn>
        </v-col>
        <v-col cols="6">
          <v-btn @click="sendImageLineareModel" block rounded color="warning">Lancer la prédiction modele lineaire</v-btn>
        </v-col>
        <v-col cols="12">
          <v-text-field
            label="Tenserflow / Keras"
            :value="tenserflowValue"
            v-model="tenserflowValue"
            outlined
            readonly
          ></v-text-field>
        </v-col>
        <v-col cols="12">
          <v-btn @click="sendImageTenserflow" block rounded color="success">Lancer la prédiction Tenserflow</v-btn>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script>
// @ is an alias to /src
import axios from 'axios'

export default {
  name: 'MachineLearning',
  data () {
    return {
      file: null,
      jsonFile: '',
      mlpValue: '',
      tenserflowValue: '',
      event: null,
      lineareModelValue: '',
      jsonFileLineareModel: '',
      eventLineareModel: null,
      items: [
        'Bas',
        'Chaussure',
        'Haut'
      ],
      selectFirstClass: 'Bas',
      selectSecondClass: 'Chaussure'
    }
  },
  methods: {
    downloadJsonFile (event) {
      console.log(event)
      const reader = new FileReader()
      reader.readAsText(event, 'UTF-8')
      reader.onload = evt => {
        this.jsonFile = evt.target.result
      }
      console.log(this.jsonFile)
    },
    downloadJsonFileModelLineaire (event) {
      console.log(event)
      const reader = new FileReader()
      reader.readAsText(event, 'UTF-8')
      reader.onload = evt => {
        this.jsonFileLineareModel = evt.target.result
      }
      console.log(this.jsonFileLineareModel)
    },
    downloadFile (event) {
      console.log(event)
      this.event = event
      if (event) {
        this.createImage(event)
      } else {
        this.file = null
      }
    },
    createImage (file) {
      // const image = new Image()
      const reader = new FileReader()
      reader.onload = (e) => {
        this.file = e.target.result
      }
      console.log(this.file)
      reader.readAsDataURL(file)
      console.log(this.file)
    },
    sendImageMLP () {
      this.mlpValue = ''
      if (this.event || this.jsonFile !== '') {
        console.log(this.event)
        const formData = new FormData()
        formData.append('image', this.event)
        axios.post('http://127.0.0.1:8000/ml/predict_image', formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }).then((result) => {
          console.log(this.mlpValue)
          console.log(result.data)
          this.mlpValue = result.data.message
        })
          .catch(function () {
            console.log('FAILURE!!')
          })
      } else {
        console.log('Image or Json file not found')
      }
    },
    sendImageTenserflow () {
      this.tenserflowValue = ''
      if (this.event) {
        console.log(this.event)
        const formData = new FormData()
        formData.append('image', this.event)
        axios.post('http://127.0.0.1:8000/ml/tf_predict', formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }).then((result) => {
          console.log(this.tenserflowValue)
          console.log(result.data)
          this.tenserflowValue = result.data.message
        })
          .catch(function () {
            console.log('FAILURE!!')
          })
      } else {
        console.log('Image or Json file not found')
      }
    },
    sendImageLineareModel () {
      this.lineareModelValue = ''
      if (this.selectFirstClass === this.selectSecondClass) {
        console.log('ERREUR CLASSE SIMILAIRE')
        return
      }
      if (this.event || this.jsonFileLineareModel !== '') {
        console.log(this.event)
        const formData = new FormData()
        formData.append('image', this.event)
        axios.post('http://127.0.0.1:8000/ml/lineare_model', formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }).then((result) => {
          console.log('test' + this.lineareModelValue)
          console.log(result.data.message)
          if (result.data.message === 1) {
            this.lineareModelValue = this.selectFirstClass
          } else {
            this.lineareModelValue = this.selectSecondClass
          }
        })
          .catch(function () {
            console.log('FAILURE!!')
          })
      } else {
        console.log('Image or Json file not found')
      }
    },
    sendJsonFile () {
      if (this.jsonFile) {
        console.log(this.jsonFile)
        axios.post('http://127.0.0.1:8000/ml/save_mlp', this.jsonFile,
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }).then((result) => {
          console.log(this.mlpValue)
          console.log(result.data)
        })
          .catch(function () {
            console.log('FAILURE!!')
          })
      } else {
        console.log('file null')
      }
    },
    sendJsonFileModeleLineaire () {
      if (this.jsonFileLineareModel) {
        console.log(this.jsonFileLineareModel)
        axios.post('http://127.0.0.1:8000/ml/json_load_modele_lineaire', this.jsonFileLineareModel,
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }).then((result) => {
          console.log(result.data)
        })
          .catch(function () {
            console.log('FAILURE!!')
          })
      } else {
        console.log('file null')
      }
    }
  }
}
</script>
