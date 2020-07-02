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
            label="Framework Machine Learning (MLP)"
            :value="mlpValue"
            v-model="mlpValue"
            outlined
            readonly
          >sdcsdv</v-text-field>
        </v-col>
        <v-col cols="6">
          <v-btn @click="sendJsonFile" block rounded color="primary">Charger le modéle MLP</v-btn>
        </v-col>
        <v-col cols="6">
          <v-btn @click="sendImage" block rounded color="primary">Lancer la prédiction MLP</v-btn>
        </v-col>
        <v-col cols="6">
          <v-file-input label="Modéle pré-entrainé Tensorflow" accept=".json" outlined dense></v-file-input>
        </v-col>
        <v-col cols="6">
          <v-text-field
            label="Framework Tensorflow"
            value=""
            outlined
            readonly
          >toto</v-text-field>
        </v-col>
        <v-col cols="6">
          <v-btn block rounded color="primary">Charger le modéle Tensorflow</v-btn>
        </v-col>
        <v-col cols="6">
          <v-btn block rounded color="primary">Lancer la prédiction Tensorflow</v-btn>
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
      event: null
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
    sendImage () {
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
    }
  }
}
</script>
