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
          <v-file-input @change="downloadFile" accept="image/*" label="File input" prepend-icon="mdi-camera"></v-file-input>
        </v-col>
        <v-col cols="6">
          <v-text-field
            label="Framework Machine Learning"
            value="Test"
            outlined
            readonly
          ></v-text-field>
        </v-col>
        <v-col cols="6">
          <v-text-field
            label="Framework Tensorflow"
            value="Test"
            outlined
            readonly
          ></v-text-field>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script>
// @ is an alias to /src
export default {
  name: 'MachineLearning',
  data () {
    return {
      file: null
    }
  },
  methods: {
    downloadFile (event) {
      console.log(event)
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
      reader.readAsDataURL(file)
    }
  }
}
</script>
